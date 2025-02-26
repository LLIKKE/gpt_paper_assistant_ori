import json
import configparser
import os
import time
from zhipuai import ZhipuAI
from openai import OpenAI
import openai
from requests import Session
from typing import TypeVar, Generator
import io
import dataclasses
from retry import retry
from tqdm import tqdm

from arxiv_scraper import get_papers_from_arxiv_rss_api
from filter_papers import filter_by_author, filter_by_gpt, filter_papers_by_hindex
from parse_json_to_md import render_md_string
from push_to_slack import push_to_slack
from arxiv_scraper import EnhancedJSONEncoder

T = TypeVar("T")


def batched(items: list[T], batch_size: int) -> list[T]:
    # takes a list and returns a list of list with batch_size
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def argsort(seq):
    # native python version of an 'argsort'
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)


def get_paper_batch(
    session: Session,
    ids: list[str],
    S2_API_KEY: str,
    fields: str = "paperId,title",
    **kwargs,
) -> list[dict]:
    # gets a batch of papers. taken from the sem scholar example.
    params = {
        "fields": fields,
        **kwargs,
    }
    if S2_API_KEY is None:
        headers = {}
    else:
        headers = {
            "X-API-KEY": S2_API_KEY,
        }
    body = {
        "ids": ids,
    }

    # https://api.semanticscholar.org/api-docs/graph#tag/Paper-Data/operation/post_graph_get_papers
    with session.post(
        "https://api.semanticscholar.org/graph/v1/paper/batch",
        params=params,
        headers=headers,
        json=body,
    ) as response:
        response.raise_for_status()
        return response.json()


def get_author_batch(
    session: Session,
    ids: list[str],
    S2_API_KEY: str,
    fields: str = "name,hIndex,citationCount",
    **kwargs,
) -> list[dict]:
    # gets a batch of authors. analogous to author batch
    params = {
        "fields": fields,
        **kwargs,
    }
    if S2_API_KEY is None:
        headers = {}
    else:
        headers = {
            "X-API-KEY": S2_API_KEY,
        }
    body = {
        "ids": ids,
    }

    with session.post(
        "https://api.semanticscholar.org/graph/v1/author/batch",
        params=params,
        headers=headers,
        json=body,
    ) as response:
        response.raise_for_status()
        return response.json()


@retry(tries=3, delay=2.0)
def get_one_author(session, author: str, S2_API_KEY: str) -> str:
    # query the right endpoint https://api.semanticscholar.org/graph/v1/author/search?query=adam+smith
    params = {"query": author, "fields": "authorId,name,hIndex", "limit": "10"}
    if S2_API_KEY is None:
        headers = {}
    else:
        headers = {
            "X-API-KEY": S2_API_KEY,
        }
    with session.get(
        "https://api.semanticscholar.org/graph/v1/author/search",
        params=params,
        headers=headers,
    ) as response:
        # try catch for errors
        try:
            response.raise_for_status()
            response_json = response.json()
            if len(response_json["data"]) >= 1:
                return response_json["data"]
            else:
                return None
        except Exception as ex:
            print("exception happened" + str(ex))
            return None


def get_papers(
    ids: list[str], S2_API_KEY: str, batch_size: int = 100, **kwargs
) -> Generator[dict, None, None]:
    # gets all papers, doing batching to avoid hitting the max paper limit.
    # use a session to reuse the same TCP connection
    with Session() as session:
        # take advantage of S2 batch paper endpoint
        for ids_batch in batched(ids, batch_size=batch_size):
            yield from get_paper_batch(session, ids_batch, S2_API_KEY, **kwargs)


def get_authors(
    all_authors: list[str], S2_API_KEY: str, batch_size: int = 100, **kwargs
):
    # first get the list of all author ids by querying by author names
    author_metadata_dict = {}
    with Session() as session:
        for author in tqdm(all_authors):
            auth_map = get_one_author(session, author, S2_API_KEY)
            if auth_map is not None:
                author_metadata_dict[author] = auth_map
            # add a 20ms wait time to avoid rate limiting
            # otherwise, semantic scholar aggressively rate limits, so do 1s
            if S2_API_KEY is not None:
                time.sleep(0.02)
            else:
                time.sleep(1.0)
    return author_metadata_dict


def get_papers_from_arxiv(config):
    area_list = config["FILTERING"]["arxiv_category"].split(",")
    paper_set = set()
    for area in area_list:
        papers = get_papers_from_arxiv_rss_api(area.strip(), config)
        paper_set.update(set(papers))
    if config["OUTPUT"].getboolean("debug_messages"):
        print("Number of papers:" + str(len(paper_set)))
    return paper_set


def parse_authors(lines):
    # parse the comma-separated author list, ignoring lines that are empty and starting with #
    author_ids = []
    authors = []
    for line in lines:
        if line.startswith("#"):
            continue
        if not line.strip():
            continue
        author_split = line.split(",")
        author_ids.append(author_split[1].strip())
        authors.append(author_split[0].strip())
    return authors, author_ids


def translate_to_chinese_via_deepseek(text: str, client: OpenAI,config) -> str:
    """
    使用 DeepSeek API 将英文文本翻译成中文
    """
    try:
        # 使用 DeepSeek API 的 chat.completions.create 方法
        response = client.chat.completions.create(
            model=config["SELECTION"]["model"],
            messages=[
                {"role": "system", "content": "You are a helpful assistant that translates English text to Chinese."},
                {"role": "user", "content": f"Translate the following text to Chinese:\n\n{text}"},
            ],
            stream=False,
            temperature=1.0,
            seed=0
        )
        # 获取返回的翻译文本
        translated_text = response.choices[0].message.content.strip()
        return translated_text
    except Exception as e:
        print(f"翻译失败: {e}")
        return text



if __name__ == "__main__":
    # now load config.ini
    # 设置 OpenAI API 密钥和 DeepSeek base_url
    
    config = configparser.ConfigParser()
    config.read("configs/config.ini")

    S2_API_KEY = os.environ.get("S2_KEY")
    OAI_KEY = config["SELECTION"]["api"]
    if OAI_KEY is None:
        raise ValueError(
            "OpenAI key is not set - please set OAI_KEY to your OpenAI key"
        )

    if "deepseek" in config["SELECTION"]["model"]:
        openai_client = OpenAI(api_key=OAI_KEY, base_url="https://api.deepseek.com")
    elif "glm" in config["SELECTION"]["model"]:
        openai_client = ZhipuAI(api_key=OAI_KEY)
    else:
        raise ValueError()

    # load the author list
    with io.open("configs/authors.txt", "r") as fopen:
        author_names, author_ids = parse_authors(fopen.readlines())
    author_id_set = set(author_ids)

    papers = list(get_papers_from_arxiv(config))
    # dump all papers for debugging

    all_authors = set()
    for paper in papers:
        all_authors.update(set(paper.authors))

    if config["OUTPUT"].getboolean("debug_messages"):
        print("Getting author info for " + str(len(all_authors)) + " authors")

    #all_authors = get_authors(list(all_authors), S2_API_KEY) # 耗时,得到作者id信息和对应hindx

    # selected_papers首先根据作者来保留一部分论文到
    #selected_papers, all_papers = filter_by_author(
    #    all_authors, papers, author_id_set, config
    #)

    all_papers = {}
    selected_papers = {}
    for paper in papers:
        all_papers[paper.arxiv_id] = paper

    # selected_papers 是根据作者选出来的论文

    filter_by_gpt(
        papers,
        config,
        openai_client,
        all_papers,
        selected_papers,
    )

    #增加翻译成中文的模块
    # 对筛选出的论文标题和摘要进行翻译
    for paper_id, paper in selected_papers.items():
        print(f"Translating paper: {paper['title']}")
        paper['title_cn'] = translate_to_chinese_via_deepseek(paper['title'], openai_client,config)
        paper['abstract_cn'] = translate_to_chinese_via_deepseek(paper['abstract'], openai_client,config)

    # sort the papers by relevance and novelty

    # pick endpoints and push the summaries
    if True:
        if config["OUTPUT"].getboolean("dump_json"):
            with open(config["OUTPUT"]["output_path"] + "output.json", "w") as outfile:
                json.dump(selected_papers, outfile, indent=4)
        if config["OUTPUT"].getboolean("dump_md"):
            with open(config["OUTPUT"]["output_path"] + "output.md", "w") as f:
                f.write(render_md_string(selected_papers))
            # 生成包含中文翻译的 Markdown 文件
            with open(config["OUTPUT"]["output_path"] + "output_translated.md", "w") as f:
                for paper_id, paper in selected_papers.items():
                    f.write(f"## {paper['title_cn']}\n\n")
                    f.write(f"{paper['abstract_cn']}\n\n")

