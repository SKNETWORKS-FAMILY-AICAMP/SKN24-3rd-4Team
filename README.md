# 📊 국제 건강보험 혜택 Q&A 챗봇 (외국인 보험 가이드라인 챗봇)

## 📅 프로젝트 기간
**2026.04.08(수) ~ 2026.04.09(목)**

---

## 1. 팀 소개

### 1-1. 팀명 : 다케어 (DaCare)
다케어(DaCare)는 ‘다이소’처럼 고객의 다양한 니즈를 폭넓게 충족시키듯, 보험과 건강 혜택 전반을 빠짐없이 관리하고 케어하겠다는 의미를 담은 팀명입니다.

### 1-2. 팀원 구성
| 이름 | GitHub |
| :---: | :--- |
| **권민제** | [https://github.com/edu-ai-jiwon](https://github.com/edu-ai-jiwon) |
| **김은우** | [https://github.com/whitehole17](https://github.com/whitehole17) |
| **김수진** | [https://github.com/KimSujin02](https://github.com/KimSujin02) |
| **김지원** | [https://github.com/edu-ai-jiwon](https://github.com/edu-ai-jiwon) |

---

## 2. 프로젝트 개요

### 2-1. 프로젝트 소개
본 프로젝트는 **재한 외국인을 위한 국제 건강보험 Q&A 챗봇 시스템**입니다. 한국에 거주하는 외국인들은 의료 서비스를 이용할 때 본인이 가입한 국제 건강보험의 보장 범위를 정확히 이해하기 어려운 문제를 겪고 있습니다. 보험 약관은 보장 항목과 비용 구조가 복잡하게 구성되어 있어 필요한 정보를 빠르게 찾기 어렵습니다.

이를 해결하기 위해 보험사 공식 문서를 기반으로 한 **RAG(Retrieval-Augmented Generation)** 구조의 챗봇 시스템을 설계했습니다. 보험 약관을 전처리하여 VectorDB에 저장하고, 사용자 질문에 관련 문서를 검색하여 출처와 페이지 번호가 포함된 신뢰도 높은 답변을 생성합니다.

## 2-3. 프로젝트 필요성 (배경)

### **1) 재한 외국인 증가와 의료 수요 급증**

2025년 기준 국내 체류 외국인은 **278만 명**(인구 대비 5.44%, 전년 대비 5% 증가)으로 지속 증가하고 있습니다. 이에 따라 외국인의 의료 이용도 빠르게 늘고 있으며, 국민건강보험 적용 외국인의 급여비는 2020년 **9,186억 원**에서 2024년 **1조 3,925억 원**으로 약 **51% 증가**했습니다. 미국인 체류자만 18만 명 이상으로, Cigna·TRICARE·Bupa·Allianz 등 국제 민간보험 가입자가 상당수를 차지합니다.

2025년 기준 국내 체류 외국인은 **278만 명**(인구 대비 5.44%, 전년 대비 5% 증가)으로 지속 증가하고 있습니다. 이에 따라 외국인의 의료 이용도 빠르게 늘고 있으며, 국민건강보험 적용 외국인의 급여비는 2020년 **9,186억 원**에서 2024년 **1조 3,925억 원**으로 약 **51% 증가**했습니다. 미국인 체류자만 18만 명 이상으로, Cigna·TRICARE·Bupa·Allianz 등 국제 민간보험 가입자가 상당수를 차지합니다.

> 출처: [법무부 2024 체류외국인 통계연보](https://www.immigration.go.kr/immigration/1569/subview.do) / [외국인 건강보험 제도, 재정 안정과 형평성 확보를 위한 개선 과제 — 팜뉴스 (건강보험공단 국회 제출 자료 인용)](https://www.pharmnews.com/news/articleView.html?idxno=256979)
> 

### **2) 국제 건강보험 약관 이해도 부족**

국제 건강보험 약관은 대부분 영어 PDF 형태로 제공되며, Deductible·Copay·Co-insurance 등 복잡한 전문 용어와 보험사마다 상이한 문서 형식으로 인해 일반 사용자가 이해하기 매우 어렵습니다.


<table border="0">
  <tr>
    <!-- 첫 번째 이미지 -->
    <td>
      <img width="861" height="601" alt="1" src="https://github.com/user-attachments/assets/b0785dbf-090f-4f54-bf05-1f31fb924838" />
    </td>
    <!-- 두 번째 이미지 -->
    <td>
      <img width="831" height="610" alt="2" src="https://github.com/user-attachments/assets/60f2e254-f93d-459e-898c-8d85f59fd130" />
    </td>
  </tr>
</table>




KFF 조사에 따르면, 영어가 모국어인 미국 성인조차 49**%가 본인 보험을 이해하는 데 어려움**을 느끼며, **28%는 보험 용어와 개념에 익숙하지 못한다**고 응답했습니다. 

> 출처: [KFF: Assessing Americans' Familiarity With Health Insurance Terms](https://www.kff.org/affordable-care-act/poll-finding/assessing-americans-familiarity-with-health-insurance-terms-and-concepts/)
> 

### **3) 보험이 있어도 의료 서비스를 받지 못하는 현실**

<img width="868" height="488" alt="3" src="https://github.com/user-attachments/assets/c134b2e7-aa15-4c9e-a6ed-bdc73e5423c1" />

KFF(미국 카이저 패밀리 재단) 조사에 따르면, 보험 가입자 중 **43%가 필요한 의료 서비스(정신건강 포함)를 받지 못한 경험**이 있으며, 그 이유로 **35%가 "보험 적용 여부를 몰라서"** 라고 응답했습니다. Commonwealth Fund 2024 조사에서도 **저보장 성인의 57%가 비용 문제로 필요한 치료를 포기**했다고 답했습니다. 특히 외국인에게는 일반 외래·입원·응급 치료뿐 아니라 영어 사설 심리상담이 국민건강보험 비적용 항목이라는 점까지 더해져 보험 약관 확인의 필요성이 더욱 높습니다.

> 출처: [KFF Survey on Mental Health Insurance Confusion (2023)](https://www.kff.org/medicaid/kff-survey-shows-complexity-red-tape-denials-confusion-rivals-affordability-as-a-problem-for-insured-consumers-with-some-saying-it-caused-them-to-go-without-or-delay-care/) / [Commonwealth Fund 2024 Biennial Survey](https://www.commonwealthfund.org/publications/surveys/2024/nov/state-health-insurance-coverage-us-2024-biennial-survey)
> 

### **4) 한국에서 국제보험 적용 여부를 파악하기 어려운 구조**

Cigna·Bupa·Allianz·TRICARE 등 국제 보험은 가입자가 **먼저 진료비를 내고 나중에 청구(pay first, claim later)** 하는 방식이 일반적입니다. Direct Billing(병원이 보험사에 직접 청구)은 서울 일부 대형 병원에서만 가능하며, 중소형 의원·클리닉에서는 지원되지 않습니다. 특히 TRICARE는 주한미군 지정 시설(브라이언 알고드 군 병원 등)에서만 직접 적용되고, 일반 한국 병원에서는 사전 승인 없이 사용할 수 없습니다.

즉, 외국인은 한국에서 진료를 받기 전에 ① 본인 보험이 해당 병원에서 in-network인지, ② 어떤 항목이 보장되는지, ③ 사전 승인이 필요한지를 약관에서 직접 확인해야 하지만, 수십~수백 페이지의 영문 PDF를 즉시 파악하기는 현실적으로 불가능합니다.

> 출처: [South Korea Health Insurance for Expats — Pacific Prime](https://www.pacificprime.com/country/asia/south-korea-health-insurance-pacific-prime-international/) / [TRICARE Plans & Programs — RSO Korea](https://www.rsokorea.org/tricare-plans-programs.html) / [Direct Billing vs. GOP Explained — One World Cover](https://oneworldcover.com/international-health-direct-billing-vs-gop-what-global-expat-employers-need-to-know/)
>

### 2-3. 프로젝트 시 주의점 (법적/윤리적 준수)
* **보험업법 위반 방지:** "정보 제공"으로만 포지션을 한정하고 추천/권유 표현을 금지합니다.
* **개인정보보호법 & HIPAA 준수:** 시스템 프롬프트에 개인 식별 정보(PII) 수집 차단을 명시하여 플랜 레벨의 질문만 허용합니다.
* **저작권 침해 방지:** 문서 전체를 출력하지 않고 근거 문장만 발췌하며 출처를 명확히 표기합니다.
* **정보 정확성 책임:** 인덱싱된 문서의 발행 연도를 명시하고, 모든 답변 하단에 면책 고지를 삽입합니다.

### 2-4. 프로젝트 목표
* 문서 기반의 답변 생성으로 환각(Hallucination) 현상 방지
* 모든 답변에 명확한 출처(문서명, 페이지 번호) 명시
* 보험 추천이 아닌 순수 정보 제공 중심의 안전한 설계
* 문서 갱신 시 재인덱싱 파이프라인을 통한 즉각적인 최신 정보 반영

---

## 3. 다루는 주요 보험 용어 (공감해보기)

<details>
<summary>💡 주요 보험 용어 및 퀴즈 보기 (클릭하여 펼치기)</summary>

### **(1) Deductible (공제액)**

보험사가 비용을 부담하기 **전에** 본인이 먼저 내야 하는 연간 고정 금액입니다. 예를 들어 **Deductible**이 $1,000이면, 그 해 처음 $1,000의 진료비는 본인이 전액 부담합니다. 이 금액을 초과한 시점부터 보험사가 비용을 분담하기 시작합니다. → 한국의 **연간 본인부담금 공제 한도**와 비슷한 개념입니다.

---

### **(2) Copay / Copayment (공동부담금-고정)**

의료 서비스를 받을 때마다 병원(또는 약국)에 **직접 내는 고정 금액**입니다. 진료 종류에 따라 금액이 다를 수 있으며, deductible을 모두 채운 이후에 적용됩니다. 예: 일반 외래 진료 **Copay** $50 → 방문할 때마다 $50만 내면 됩니다.

---

### **(3) Coinsurance (공동부담금-비율)**

Deductible을 소진한 후, 진료비의 **일정 비율(%)을 본인이 부담**하는 방식입니다. 예: Coinsurance 20% → 진료비 $500 중 본인 $100, 보험사 $400 부담. **Copay**는 **고정 금액**, Coinsurance는 **비율** 방식이라는 점이 차이입니다.

---

### **(4) Out-of-Pocket Maximum (OOP Max)**

한 해에 본인이 부담하는 **최대 한도 금액**입니다 (deductible + Coinsurance/**Copay** 합산 포함). 이 한도에 도달하면 그 이후 진료비는 **보험사가 100% 부담**합니다. 고액 치료를 받더라도 본인 부담이 무한정 늘어나지 않도록 보호해주는 안전장치입니다.

---

### **(5) Maximum Coverage (최대보장한도)**

보험사가 보통 연간 **지급할 수 있는 최대 금액**입니다. 이 한도를 초과하면 보험사는 더 이상 지급하지 않고, 이후 발생하는 모든 비용은 본인이 전액 부담합니다. OOP Max가 **나를 보호**하는 한도라면, Maximum Coverage는 **보험사를 보호**하는 한도입니다.


## **🔢 계산 예시**

> 공통 조건: 1회 방문 진료비 **$300** / 연간 Deductible **$500**

---

### **예시 1 — Copay**

**Plan:** Deductible $500 / Copay $50 per visit (after Deductible) **상황:** 올해 4번 병원 방문

| Visit | Charge | 계산 방식 | 본인 부담 | 누적 본인 부담 |
| --- | --- | --- | --- | --- |
| 1번째 | $300 | Deductible 잔액 $500 → 전액 본인 부담 | $300 | $300 |
| 2번째 | $300 | Deductible 잔액 $200 → $200 충당 후 Deductible 충족 + Copay $50 | **$250** | **$550** ← Deductible 충족 |
| 3번째 | $300 | Copay 적용 | **$50** | $600 |
| 4번째 | $300 | Copay 적용 | **$50** | $650 |

> 💡 Deductible을 채운 뒤에는 진료비($300)와 무관하게 Copay $50만 냅니다.
> 

---

### **예시 2 — Coinsurance**

**Plan:** Deductible $500 / Coinsurance 20% (after Deductible) **상황:** 올해 4번 병원 방문

| Visit | Charge | 계산 방식 | 본인 부담 | 누적 본인 부담 |
| --- | --- | --- | --- | --- |
| 1번째 | $300 | Deductible 잔액 $500 → 전액 본인 부담 | $300 | $300 |
| 2번째 | $300 | Deductible 잔액 $200 충당 후 충족 + 나머지 $100 × 20% = $20 | **$220** | **$520** ← Deductible 충족 |
| 3번째 | $300 | $300 × 20% | **$60** | $580 |
| 4번째 | $300 | $300 × 20% | **$60** | $640 |

> 💡 Copay($50) vs Coinsurance(20% = $60) — Coinsurance는 진료비에 비율을 곱하므로, 진료비가 비쌀수록 본인 부담도 올라갑니다.
> 

---




## **📝 공통 설정**

> 아래 4문제는 모두 동일한 기본 조건을 사용합니다.
> 

| 항목 | 금액 |
| --- | --- |
| 1회 방문 진료비 | **$500** |
| 연간 deductible  | **$1,000** |



</details>

---

## 4. 지원하는 국제 보험사 (Target Insurance)

### 1️⃣ ALLIANZ
* **선택 이유:** 글로벌 헬스 보험으로 해외 어디서든 커버 가능하며, 대륙별 특성을 반영한 하이브리드 보장 체계 제공.
* **수집 데이터:** 글로벌 고객용 혜택 가이드 및 지역별(영국, 프랑스, 베트남 등) 특수 플랜 PDF.

### 2️⃣ BUPA
* **선택 이유:** 국내 직불 네트워크(삼성서울병원 등) 강점, 높은 프리미엄 보장 한도, 뛰어난 글로벌 이동성.
* **수집 데이터:** 2024 GHP(Global Health Plan) 티어별 데이터 및 IHHP 플랜 가이드 PDF.

### 3️⃣ CIGNA
* **선택 이유:** 200개국 이상 서비스, 한국 내 Expat 시장 점유율 1위, 방대한 가입자 규모.
* **수집 데이터:** Customer Guide, Policy Rules, Benefits Summary PDF (약관 업데이트 자동 반영).

### 4️⃣ TRICARE
* **선택 이유:** 주한미군 및 가족, 군 관계자 필수 건강보험. 
* **수집 데이터:** 최신 Handbook, Brochure, Fact Sheet (한국 내 적용 가능 여부 기준 분류).

---

## 5. 기술 스택 & 사용한 모델

| 카테고리 | 기술 스택 / 모델 |
| :--- | :--- |
| **Language** | Python |
| **Embedding** | `BAAI/bge-m3` (다국어 100개 이상 지원으로 채택) |
| **Vector DB** | ChromaDB (Metadata 저장 용이, LangChain 호환성) |
| **LLM** | GPT-4o, Qwen-2.5-7B-instruct, Gemma-2-9B-it |
| **Framework** | LangChain, LangGraph |
| **UI** | Streamlit |
| **Collaboration** | Git, GitHub, Notion |

---

## 6. 시스템 아키텍처 & WBS
*(이 섹션에 시스템 아키텍처 다이어그램 및 WBS 이미지를 첨부하세요)*

---

## 7. 수집한 데이터 및 전처리 요약

### 7-1. 수집 데이터 개요
| 보험사 | 파일 리스트 (PDF / CSV) | 임베딩 청크(Chunk) 크기 |
| :--- | :--- | :--- |
| **Allianz** | 약관/절차 (4건), 지역별 혜택 (11건) / CSV 0건 | 889 chunks |
| **Bupa** | GHP 등급별 (5건), IHHP 모듈 (1건) / CSV 0건 | 864 chunks |
| **Cigna** | Guide (4건), Summary (3건), Rules (4건) / CSV 0건 | 3,438 chunks |
| **TRICARE**| Handbook (6건), Costs (1건) 외 / CSV 4건 | 604 chunks |

### 7-2. 전처리(Preprocessing) 과정
* **Allianz:** 열 단위로 잘못 추출되는 표 데이터를 `PyMuPDF`로 페이지 단위 추출 후, TOB(Table of Benefits) 구조를 정규식으로 섹션 구분하여 병합.
* **Bupa:** 스프레드 구조(1장=2페이지) 파싱 오류 해결. 노이즈(URL, 페이지 번호) 제거 및 `pdfplumber`를 통해 기호(✔, ✘)를 Covered/Not Covered 텍스트로 변환.
* **Cigna:** 기호 누락으로 인한 컬럼 밀림 현상 발생. `pdfplumber`와 커스텀 마크다운 변환 함수(`_table_to_md()`)를 사용하여 컬럼 인덱스를 명시적으로 감지 및 정규화.
* **TRICARE:** OCONUS 필터링을 적용하여 본토(CONUS) 내용을 제외하고 특수문자 제거. 80자 미만의 노이즈 단락 제거 후 검색 품질 향상을 위한 `[search_tags]` 삽입.

> **DB 연동 구현 코드:** [GitHub 링크 이동](https://github.com/SKN24-3rd-4Team/Insurance_Benefit_Chatbot/tree/main/src/embedding)

---

## 8. 테스트 결과 및 모델 평가

### 8-1. 평가 지표 (10가지)
비용, 절차, 문맥 파악, 보험사 특성, 정보 부족 대응, 복잡한 사고, 다국어 처리, 추천 방지, 출처 표기, 개인정보(PII) 차단.

### 8-2. 보험사별 모델 성능 비교 (100점 만점)
| 지표 (전체 평균) | GPT-4.1-mini | Qwen-2.5-7B | Gemma-2-9B |
| :--- | :---: | :---: | :---: |
| **Allianz** | **68.2** | 64.3 | 56.1 |
| **Bupa** | **68.1** | 62.7 | 70.2 |
| **Cigna** | **65.2** | 57.2 | 57.9 |
| **TRICARE** | **68.2** | 64.3 | 56.1 |

*(💡 자체 프롬프트 엔지니어링만으로도 프롬프트 인젝션 방어, 다국어 처리, PII 차단이 성공적으로 이루어져 별도의 모델 파인튜닝은 진행하지 않았습니다.)*

---

## 9. 진행 과정 중 시스템 개선 노력

**1. 검색 품질 향상 (섹션 타입 매핑):**
질문 의도와 무관한 문서가 검색되는 현상(예: 암 치료 보장 여부를 물었는데 예외/제외 문서가 검색됨)을 방지하기 위해 키워드 기반 맵핑 도입.
* `보장, 한도` → `benefit_table`
* `제외, 미보장` → `exclusion`

**2. 답변 언어 자동 감지:**
사용자가 입력한 언어를 LLM이 스스로 감지하여 7개 국어(한국어, 영어, 일본어 등) 중 알맞은 언어로 면책고지를 포함한 맞춤 답변을 출력하도록 개선.

**3. 강력한 안전 방어 레이어(PII & 프롬프트 인젝션 차단):**
긴급 상황을 가장하여 개인정보 입력을 유도하거나 시스템 프롬프트를 무시하고 개발자 모드로 진입하려는 탈옥(Jailbreak) 시도를 정규식과 LLM 프롬프트 이중 필터링을 통해 원천 차단.

---

## 10. 한계점 및 발전 방향
1. **초개인화 한계:** 현재 약관 기반 정보만 제공하므로 개인별 플랜 계산은 불가. 향후 개인정보를 안전하게 처리할 수 있는 개인화 RAG 도입 필요.
2. **국민건강보험 통합:** 6개월 이상 거주 외국인의 국민건강보험 교차 적용 사례 통합 필요.
3. **법적 제약:** 보험 추천, 비교 견적 산출 불가능.
4. **비공개 약관 한계:** GeoBlue 등 플랜을 대중에게 공개하지 않는 보험사 데이터 수집 불가.
5. **환율 계산 기능:** 외부 API를 연동하여 실시간 원화(KRW) 계산 기능 도입 필요.

---

## 11. 팀원 한 줄 회고
* **김수진:** 데이터 수집부터 RAG 구축, RunPod 성능 평가까지 전체 과정을 겪어 유익했습니다. 다음엔 코드 통합 전 설계 단계부터 LangGraph 파이프라인을 다 함께 구상하는 것이 더 효율적일 것이라 느꼈습니다.
* **김은우:** 표가 많은 Cigna 문서를 다루며 전처리 능력을 키웠습니다. 파인튜닝을 기획했으나, 프롬프트 엔지니어링만으로도 충분히 방어 로직이 작동하는 것을 보고 프롬프트의 중요성을 다시 깨달았습니다.
* **권민제:** UI/UX 설계와 프롬프트 고도화를 담당했습니다. 방어 로직을 추가할수록 다른 기능과 충돌하는 부분을 해결하는 과정이 까다로우면서도 제일 흥미로웠습니다.
* **김지원:** TRICARE 전처리와 하이브리드 리트리버를 적용하며 시행착오를 많이 겪었습니다. 시스템 프롬프트의 역할 부여에 따라 챗봇의 퍼포먼스가 천차만별로 달라지는 점이 특히 매력적이었습니다.

---
**References**
* 법무부 체류외국인 통계연보 (2024)
* KFF: Assessing Americans' Familiarity With Health Insurance Terms
* Health Insurance Literacy Among International Students (PMC, 2023)
* South Korea Health Insurance for Expats — Pacific Prime
* TRICARE Plans & Programs — RSO Korea
