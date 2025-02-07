import configparser
import dataclasses
import json
from datetime import datetime, timedelta
from html import unescape
from typing import List, Optional, Tuple
import re
import arxiv

import feedparser
from dataclasses import dataclass
import logging
import os


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


@dataclass
class Paper:
    # 记录作者列表、论文标题、摘要、arxiv id
    authors: List[str]
    title: str
    abstract: str
    arxiv_id: str

    # 添加哈希函数以支持去重
    def __hash__(self):
        return hash(self.arxiv_id)


def is_earlier(ts1: str, ts2: str) -> bool:
    """
    比较两个 arxiv_id，判断 ts1 是否早于 ts2。
    arxiv_id 格式通常为 'XXXX.XXXXXvX'，通过去除点和版本号进行比较。
    """
    try:
        id1_num = int(ts1.split('v')[0].replace('.', ''))
        id2_num = int(ts2.split('v')[0].replace('.', ''))
        return id1_num < id2_num
    except:
        return False


def get_papers_from_arxiv_api(area: str, timestamp: Optional[datetime], last_id: Optional[str], intervals=7) -> List[
    Paper]:
    """
    7天内arxiv更新的最多200篇论文
    通过 arxiv API 获取指定分类的论文。
    仅包含分类为 'cs.AI' 或 'cs.LG' 的论文。
    area：所述领域
    timestamp：当前的时间
    last_id：如果last_id!=None说明前一天找到了这个领域论文，这时候虽然这个函数搜索了7天内，但是早于last_id的都不记录？？有点重复了，进入这个函数last_id==None
    """
    end_date = timestamp if timestamp else datetime.utcnow()
    start_date = end_date - timedelta(days=intervals)  # 可根据需要调整时间范围

    # 构造查询语句，确保只包含指定分类
    query = f"cat:{area} AND submittedDate:[{start_date.strftime('%Y%m%d')} TO {end_date.strftime('%Y%m%d')}]"
    search = arxiv.Search(
        query=query,
        max_results=200,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    results = search.results()
    api_papers = []
    for result in results:
        new_id = result.get_short_id()
        # 如果设置了 last_id，跳过早于 last_id 的论文
        if last_id and is_earlier(last_id, new_id):
            continue
        authors = [author.name for author in result.authors]
        summary = result.summary
        summary = unescape(re.sub("\n", " ", summary))  # 除换行符，并使用 unescape 将 HTML 实体转换为正常字符。
        # 确保论文属于所需的分类
        if area in result.categories:
            paper = Paper(
                authors=authors,
                title=result.title,
                abstract=summary,
                arxiv_id=new_id,
            )
            api_papers.append(paper)
    return api_papers


def get_papers_from_arxiv_rss(area: str, config: Optional[dict]) -> Tuple[
    List[Paper], Optional[datetime], Optional[str]]:
    """
    通过 arxiv RSS 获取指定分类的论文。
    仅包含分类为 'cs.AI' 或 'cs.LG' 的论文。
    """
    updated = datetime.utcnow() - timedelta(days=1)  # 计算一天的时间差
    updated_string = updated.strftime("%a, %d %b %Y %H:%M:%S GMT")
    feed = feedparser.parse(
        f"https://export.arxiv.org/rss/{area}",  # 使用 HTTPS
        modified=updated_string
    )  # modified 参数用于向服务器传递上一次的更新时间（updated_string）
    #print(feed)
    logging.info(f"Feed Status: {feed.status}")
    logging.info(f"Feed entries count: {len(feed.entries)}")

    if feed.status == 304:
        if config and config.get("OUTPUT", {}).get("debug_messages", False):
            logging.info(f"No new papers since {updated_string} for {area}")
        return [], None, None

    entries = feed.entries
    if len(entries) == 0:
        logging.info(f"No entries found for {area}")
        return [], None, None

    last_id = entries[0].link.split("/")[-1]

    # 解析 'updated' 字段
    updated_field = feed.feed.get('updated', None)
    if updated_field:
        logging.info(f"Updated field from feed: {updated_field}")
        try:
            timestamp = datetime.strptime(updated_field, "%a, %d %b %Y %H:%M:%S +0000")
            logging.info(f"Parsed timestamp: {timestamp}")
        except Exception as e:
            logging.error(f"Error parsing timestamp: {e}")
            timestamp = None
    else:
        logging.warning("No updated field found in feed.")
        timestamp = None

    paper_list = []
    for paper in entries:
        # 仅处理新论文
        if paper.get("arxiv_announce_type", "") != "new":
            continue
        # 提取分类
        paper_area = paper.tags[0].term if 'tags' in paper and len(paper.tags) > 0 else ""
        # 根据配置过滤非主分类的论文
        # if (area != paper_area) and (config and config.get("FILTERING", {}).getboolean("force_primary", False)):
        #     logging.info(f"Ignoring {paper.title} as it belongs to {paper_area} instead of {area}")
        #     continue
        # 获取 force_primary 配置项，处理为布尔值
        force_primary_value = config.get("FILTERING", "force_primary", fallback="false").strip().lower()

        # 判断 force_primary 是否为 "true"
        force_primary = force_primary_value == "true"

        # 如果是 force_primary 且区域不同，则跳过当前论文
        if area != paper_area and force_primary:
            logging.info(f"Ignoring {paper.title} as it belongs to {paper_area} instead of {area}")
            continue
        # 提取作者，去除 HTML 标签
        authors = [
            unescape(re.sub("<[^<]+?>", "", author)).strip()
            for author in paper.author.replace("\n", ", ").split(",")
        ]
        # 提取摘要，去除 HTML 标签
        summary = re.sub("<[^<]+?>", "", paper.summary)
        summary = unescape(re.sub("\n", " ", summary))
        # 提取标题，去除最后的 arxiv 信息
        title = re.sub(r"\(arXiv:[0-9]+\.[0-9]+v[0-9]+ \[.*\]\)$", "", paper.title).strip()
        # 提取 arxiv_id
        id = paper.link.split("/")[-1]
        # 确保论文属于所需的分类
        if area in paper_area:
            new_paper = Paper(authors=authors, title=title, abstract=summary, arxiv_id=id)
            paper_list.append(new_paper)
        else:
            logging.debug(f"Skipping paper {title} as it does not belong to {area}")
    return paper_list, timestamp, last_id


def merge_paper_list(paper_list: List[Paper], api_paper_list: List[Paper]) -> List[Paper]:
    """
    合并 RSS 和 API 获取的论文列表，去除重复项。
    """
    api_set = set([paper.arxiv_id for paper in api_paper_list])
    merged_paper_list = api_paper_list.copy()
    for paper in paper_list:
        if paper.arxiv_id not in api_set:
            merged_paper_list.append(paper)
    return merged_paper_list


def get_papers_from_arxiv_rss_api(area: str, config: Optional[dict]) -> List[Paper]:
    """
    综合使用 RSS 和 API 获取指定分类的论文。
    """
    paper_list, timestamp, last_id = get_papers_from_arxiv_rss(area, config)  # 爬取1天前的该领域论文
    # print(paper_list)
    # print(timestamp)
    # print(last_id)
    intervals = 3
    #paper_list = []
    print("today paper", len(paper_list))
    if len(paper_list) == 0:  # 如果没有，则扩大范围
        logging.info(f"Attempting to fetch papers from API for {area}...")
        # print(f"Attempting to fetch papers from API for {area}...")
        api_paper_list = get_papers_from_arxiv_api(area, timestamp, last_id, intervals)  # 7天内的更新
        if len(api_paper_list) == 0:  # 如果7天内还没有文章，则从7天前再数7天
            logging.info(f"No papers found via API for {area}. Trying wider search options...")
            # 例如：扩大时间范围
            if timestamp:
                extended_timestamp = timestamp - timedelta(days=intervals)
            else:
                extended_timestamp = None
            api_paper_list = get_papers_from_arxiv_api(area, extended_timestamp, last_id, intervals)
        paper_list = api_paper_list

    return paper_list


def get_papers(config: dict) -> List[Paper]:
    """
    获取所有指定分类的论文。
    """
    area_list = ["cs.AI", "cs.LG"]  # 使用具体分类，避免通配符
    all_papers = []
    for area in area_list:
        papers = get_papers_from_arxiv_rss_api(area, config)
        all_papers.extend(papers)
    return all_papers


def save_papers(papers: List[Paper], output_path: str):
    """
    将获取到的论文列表保存为 JSON 文件。
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_file = os.path.join(output_path, "papers.json")
    with open(output_file, "w", encoding='utf-8') as outfile:
        json.dump([dataclasses.asdict(paper) for paper in papers], outfile, indent=4, ensure_ascii=False,
                  cls=EnhancedJSONEncoder)


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 读取配置文件
    config = configparser.ConfigParser()
    config.read("configs/config.ini")

    # 获取论文
    all_papers = get_papers(config)

    # 保存到文件
    save_papers(all_papers, "./output/")

    # 打印摘要信息
    print(f"Fetched {len(all_papers)} papers.")

    if len(all_papers) > 0:
        # 收集所有分类
        all_categories = set()
        for paper in all_papers:
            # 由于原始Paper类没有categories字段，这里只能根据area_list推断
            # 如果需要更精确的分类，请考虑在Paper类中添加categories字段
            pass  # 跳过分类收集，因为原始Paper没有分类信息

        # 如果您需要打印分类信息，建议在Paper类中添加categories字段
        # 此处仅打印前十篇论文的标题
        print("\nTitles of the first ten papers:")
        for idx, paper in enumerate(all_papers[:10], 1):
            print(f"{idx}. {paper.title}")
    else:
        print("No papers fetched.")




