"""OpenDART 클라이언트: 디스크 캐시 + 재시도 + corp_code 매핑.

모든 응답은 CACHE_DIR에 json으로 저장되어 재실행 시 API를 다시 때리지 않는다.
(전수 수집은 수만 콜 규모라 캐시가 없으면 일 한도에 걸린다)
"""
from __future__ import annotations

import hashlib
import io
import json
import time
import zipfile
import xml.etree.ElementTree as ET

import pandas as pd
import requests

from . import config as C


class DartError(RuntimeError):
    pass


def _cache_path(endpoint: str, params: dict) -> "C.Path":
    key = json.dumps({"e": endpoint, "p": params}, sort_keys=True, ensure_ascii=False)
    h = hashlib.md5(key.encode()).hexdigest()
    return C.CACHE_DIR / f"{endpoint}_{h}.json"


def get_json(endpoint: str, use_cache: bool = True, **params) -> dict:
    """opendart.fss.or.kr/api/{endpoint}.json 호출.

    status '000' 정상, '013' 조회결과 없음(빈 dict 반환), 그 외 예외.
    """
    cp = _cache_path(endpoint, params)
    if use_cache and cp.exists():
        return json.loads(cp.read_text(encoding="utf-8"))

    url = f"{C.DART_BASE}/{endpoint}.json"
    q = {"crtfc_key": C.DART_API_KEY, **params}
    last_err: Exception | None = None
    for attempt in range(C.DART_MAX_RETRY):
        try:
            r = requests.get(url, params=q, timeout=30)
            r.raise_for_status()
            data = r.json()
            status = data.get("status")
            if status == "000":
                cp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
                time.sleep(C.DART_SLEEP)
                return data
            if status == "013":                     # 조회 결과 없음
                cp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
                time.sleep(C.DART_SLEEP)
                return data
            if status == "020":                     # 사용한도 초과 → 대기 후 재시도
                time.sleep(60)
                continue
            raise DartError(f"{endpoint} status={status} msg={data.get('message')}")
        except (requests.RequestException, ValueError) as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise DartError(f"{endpoint} 재시도 초과: {last_err}")


def get_document_xml(rcept_no: str, use_cache: bool = True) -> str:
    """공시 원문(document.xml) 전문 텍스트. 콜/풋 조항 파싱용."""
    cp = C.CACHE_DIR / f"doc_{rcept_no}.txt"
    if use_cache and cp.exists():
        return cp.read_text(encoding="utf-8", errors="ignore")
    url = f"{C.DART_BASE}/document.xml"
    r = requests.get(url, params={"crtfc_key": C.DART_API_KEY, "rcept_no": rcept_no},
                     timeout=60)
    r.raise_for_status()
    text = ""
    try:
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            parts = []
            for name in zf.namelist():
                raw = zf.read(name)
                for enc in ("utf-8", "cp949", "euc-kr"):
                    try:
                        parts.append(raw.decode(enc))
                        break
                    except UnicodeDecodeError:
                        continue
            text = "\n".join(parts)
    except zipfile.BadZipFile:
        text = r.content.decode("utf-8", errors="ignore")
    cp.write_text(text, encoding="utf-8")
    time.sleep(C.DART_SLEEP)
    return text


def load_corp_map(use_cache: bool = True) -> pd.DataFrame:
    """corpCode.xml → (corp_code, corp_name, stock_code). 상장사만 반환."""
    cp = C.CACHE_DIR / "corp_map.parquet"
    if use_cache and cp.exists():
        return pd.read_parquet(cp)
    r = requests.get(f"{C.DART_BASE}/corpCode.xml",
                     params={"crtfc_key": C.DART_API_KEY}, timeout=60)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        xml = zf.read(zf.namelist()[0])
    root = ET.fromstring(xml)
    rows = []
    for el in root.iter("list"):
        stock = (el.findtext("stock_code") or "").strip()
        if not stock:
            continue                                 # 비상장 제외
        rows.append({
            "corp_code": el.findtext("corp_code"),
            "corp_name": el.findtext("corp_name"),
            "stock_code": stock,
        })
    df = pd.DataFrame(rows).drop_duplicates("corp_code")
    df.to_parquet(cp, index=False)
    return df
