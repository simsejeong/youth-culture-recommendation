# 데이터 안내

이 폴더의 데이터는 저장소에 포함하지 않습니다.

| 구분 | 위치 | 비고 |
|---|---|---|
| 설문 원본 | `data/survey_google.xlsx`, `data/survey_modu.xlsx` | 응답자 개인정보 보호를 위해 비공개 |
| 공공데이터 원본 | `data/raw/` | 공공데이터포털 · 문화빅데이터플랫폼 · 서울 열린데이터광장 · 산림청 · 한국문화예술교육진흥원 등에서 내려받기 |
| 마스터 DB | `data/recommend_master_db.csv` | `notebooks/02_master_db_build.ipynb` 실행 결과 |

파일명은 `notebooks/02_master_db_build.ipynb`의 `build_master_db()` 안 `loaders` 목록을 기준으로 맞춰 주세요.
