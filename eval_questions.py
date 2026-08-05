"""
评估测试集：40 个问题（15 英文 / 15 中文 / 10 中英混合），每题带标准答案。

每个问题标注：
- lang:              en / zh / mixed
- question:          原始检索查询，覆盖四种表达形式：
                      简短关键词（如 "internship period"）、完整问句、
                      同义表达（如 working hours / 上班时间）、
                      中文询问英文资料、中英混合询问
- expected_source:   标准答案所在的来源文件
- expected_fragment: 标准答案段落中的唯一子串，用于自动判定检索是否命中
- rewritten_en:      中文/混合问题的独立英文改写（方案 C 专用）。
                     只做自然语言翻译，不复制 expected_fragment、不泄露标准答案。
                     英文问题不需要改写（检索时直接用原问题）。

评估语料 = 当前 chroma_data 里的全部 chunk（只读导出），与线上索引内容一致。
cloud_doc.txt 只有一句话（"Another test document about cloud computing."），
不足以构造有意义的检索问题，故未覆盖。

注意：fragment 必须与语料中的原文逐字符一致
（如 "Taylor’s University" 用的是右单引号 ’，时间行用的是短横线 –）。
fragment 的接地与唯一性由 test_eval.py 自动校验，改写字段的合规性同样由测试校验。

EXTRA_QUESTIONS：切块策略实验（eval_chunking.py）专用的失败类型补充问题，
10 题（5 英文 / 5 中文），结构与 QUESTIONS 相同，独立于 40 题基线；
接地与唯一性由 test_chunking.py 自动校验。
"""

QUESTIONS = [
    # ---------------- 英文（15） ----------------
    {
        "lang": "en",
        "question": "internship period",
        "expected_source": "university letter concerning d internship.pdf",
        "expected_fragment": "16 WEEKS OF COMPULSORY INTERNSHIP",
    },
    {
        "lang": "en",
        "question": "When does the internship start?",
        "expected_source": "university letter concerning d internship.pdf",
        "expected_fragment": "START DATE : 14 SEPTEMBER 2026",
    },
    {
        "lang": "en",
        "question": "When does the internship end?",
        "expected_source": "university letter concerning d internship.pdf",
        "expected_fragment": "END DATE : 8 JANUARY 2027",
    },
    {
        "lang": "en",
        "question": "What programme is the student enrolled in?",
        "expected_source": "university letter concerning d internship.pdf",
        "expected_fragment": "Bachelor of Information Technology",
    },
    {
        "lang": "en",
        "question": "student name",
        "expected_source": "university letter concerning d internship.pdf",
        "expected_fragment": "WU ZHONGHENG",
    },
    {
        "lang": "en",
        "question": "Which university does the student attend?",
        "expected_source": "university letter concerning d internship.pdf",
        "expected_fragment": "Taylor’s University",
    },
    {
        "lang": "en",
        "question": "full-time student",
        "expected_source": "university letter concerning d internship.pdf",
        "expected_fragment": "full-time student",
    },
    {
        "lang": "en",
        "question": "What is the objective of the industrial training?",
        "expected_source": "university letter concerning d internship.pdf",
        "expected_fragment": "hands on experience",
    },
    {
        "lang": "en",
        "question": "What does the university ask the company to send back?",
        "expected_source": "university letter concerning d internship.pdf",
        "expected_fragment": "letter of confirmation",
    },
    {
        "lang": "en",
        "question": "Who is the Head of Career Services?",
        "expected_source": "university letter concerning d internship.pdf",
        "expected_fragment": "RAJA EDRIANA BAIZURA",
    },
    {
        "lang": "en",
        "question": "Which company is offering the internship?",
        "expected_source": "Letter of Appointment Internship - WU ZHONGHENG.pdf",
        "expected_fragment": "AURAPLEX SDN. BHD.",
    },
    {
        "lang": "en",
        "question": "monthly allowance",
        "expected_source": "Letter of Appointment Internship - WU ZHONGHENG.pdf",
        "expected_fragment": "RM1,000.00",
    },
    {
        "lang": "en",
        "question": "What are the working hours?",
        "expected_source": "Letter of Appointment Internship - WU ZHONGHENG.pdf",
        "expected_fragment": "Monday – Friday 9am to 6pm",
    },
    {
        "lang": "en",
        "question": "Who is the head of the HR department?",
        "expected_source": "Letter of Appointment Internship - WU ZHONGHENG.pdf",
        "expected_fragment": "Chan Chao Jian",
    },
    {
        "lang": "en",
        "question": "What must be returned when the internship ends?",
        "expected_source": "Letter of Appointment Internship - WU ZHONGHENG.pdf",
        "expected_fragment": "return documents, equipment, and all property",
    },
    # ---------------- 中文（15） ----------------
    {
        "lang": "zh",
        "question": "实习期多长？",
        "rewritten_en": "How long is the internship?",
        "expected_source": "university letter concerning d internship.pdf",
        "expected_fragment": "16 WEEKS",
    },
    {
        "lang": "zh",
        "question": "实习什么时候开始？",
        "rewritten_en": "When does the internship start?",
        "expected_source": "university letter concerning d internship.pdf",
        "expected_fragment": "START DATE",
    },
    {
        "lang": "zh",
        "question": "实习什么时候结束？",
        "rewritten_en": "When does the internship end?",
        "expected_source": "university letter concerning d internship.pdf",
        "expected_fragment": "END DATE",
    },
    {
        "lang": "zh",
        "question": "实习生叫什么名字？",
        "rewritten_en": "What is the intern's name?",
        "expected_source": "university letter concerning d internship.pdf",
        "expected_fragment": "WU ZHONGHENG",
    },
    {
        "lang": "zh",
        "question": "学生读什么专业？",
        "rewritten_en": "What is the student's major?",
        "expected_source": "university letter concerning d internship.pdf",
        "expected_fragment": "Bachelor of Information Technology",
    },
    {
        "lang": "zh",
        "question": "学生来自哪所大学？",
        "rewritten_en": "Which university is the student from?",
        "expected_source": "university letter concerning d internship.pdf",
        "expected_fragment": "Taylor’s University",
    },
    {
        "lang": "zh",
        "question": "为什么学生必须参加实习？",
        "rewritten_en": "Why does the student have to do an internship?",
        "expected_source": "university letter concerning d internship.pdf",
        "expected_fragment": "required to undergo",
    },
    {
        "lang": "zh",
        "question": "实习培训的目的是什么？",
        "rewritten_en": "What is the purpose of the internship training?",
        "expected_source": "university letter concerning d internship.pdf",
        "expected_fragment": "real working environment",
    },
    {
        "lang": "zh",
        "question": "公司需要回传什么文件？",
        "rewritten_en": "What document should the company send back?",
        "expected_source": "university letter concerning d internship.pdf",
        "expected_fragment": "letter of confirmation",
    },
    {
        "lang": "zh",
        "question": "实习津贴是多少？",
        "rewritten_en": "How much is the internship allowance?",
        "expected_source": "Letter of Appointment Internship - WU ZHONGHENG.pdf",
        "expected_fragment": "RM1,000.00",
    },
    {
        "lang": "zh",
        "question": "上班时间是几点？",
        "rewritten_en": "What are the working hours?",
        "expected_source": "Letter of Appointment Internship - WU ZHONGHENG.pdf",
        "expected_fragment": "Monday – Friday 9am to 6pm",
    },
    {
        "lang": "zh",
        "question": "公司叫什么名字？",
        "rewritten_en": "What is the name of the company?",
        "expected_source": "Letter of Appointment Internship - WU ZHONGHENG.pdf",
        "expected_fragment": "AURAPLEX SDN. BHD.",
    },
    {
        "lang": "zh",
        "question": "实习生向谁汇报工作？",
        "rewritten_en": "Who does the intern report to?",
        "expected_source": "Letter of Appointment Internship - WU ZHONGHENG.pdf",
        "expected_fragment": "Khor Kai Dat",
    },
    {
        "lang": "zh",
        "question": "机密信息要怎么处理？",
        "rewritten_en": "How should the intern handle confidential information?",
        "expected_source": "Letter of Appointment Internship - WU ZHONGHENG.pdf",
        "expected_fragment": "strictly confidential",
    },
    {
        "lang": "zh",
        "question": "工资最晚几号之前发？",
        "rewritten_en": "When is the salary paid?",
        "expected_source": "Letter of Appointment Internship - WU ZHONGHENG.pdf",
        "expected_fragment": "shall be paid not later than",
    },
    # ---------------- 中英混合（10） ----------------
    {
        "lang": "mixed",
        "question": "实习从什么时候开始，什么时候结束？",
        "rewritten_en": "When does the internship start and end?",
        "expected_source": "Letter of Appointment Internship - WU ZHONGHENG.pdf",
        "expected_fragment": "14th September 2026 to 08th January 2027",
    },
    {
        "lang": "mixed",
        "question": "WU ZHONGHENG 的 Student ID 是多少？",
        "rewritten_en": "What is the student number of WU ZHONGHENG?",
        "expected_source": "university letter concerning d internship.pdf",
        "expected_fragment": "Student ID : Student ID : 0365042",
    },
    {
        "lang": "mixed",
        "question": "实习生的 Lunch hour 是几点到几点？",
        "rewritten_en": "When is the lunch break during the internship?",
        "expected_source": "Letter of Appointment Internship - WU ZHONGHENG.pdf",
        "expected_fragment": "Lunch hour 12pm to 1pm",
    },
    {
        "lang": "mixed",
        "question": "公司的办公地址在哪里？",
        "rewritten_en": "Where is the company located?",
        "expected_source": "Letter of Appointment Internship - WU ZHONGHENG.pdf",
        "expected_fragment": "NO.5, JALAN BS9/7B",
    },
    {
        "lang": "mixed",
        "question": "HR 部门的负责人是谁？",
        "rewritten_en": "Who is in charge of the HR department?",
        "expected_source": "Letter of Appointment Internship - WU ZHONGHENG.pdf",
        "expected_fragment": "Chan Chao Jian",
    },
    {
        "lang": "mixed",
        "question": "实习生的 job title 是什么？",
        "rewritten_en": "What is the intern's job title?",
        "expected_source": "Letter of Appointment Internship - WU ZHONGHENG.pdf",
        "expected_fragment": "AI Programmer",
    },
    {
        "lang": "mixed",
        "question": "实习津贴是 RM 多少？",
        "rewritten_en": "How much is the allowance in RM?",
        "expected_source": "Letter of Appointment Internship - WU ZHONGHENG.pdf",
        "expected_fragment": "RM1,000.00",
    },
    {
        "lang": "mixed",
        "question": "实习生要 develop 什么？",
        "rewritten_en": "What will the intern develop?",
        "expected_source": "Letter of Appointment Internship - WU ZHONGHENG.pdf",
        "expected_fragment": "AI Server",
    },
    {
        "lang": "mixed",
        "question": "公司全称是什么？",
        "rewritten_en": "What is the full name of the company?",
        "expected_source": "Letter of Appointment Internship - WU ZHONGHENG.pdf",
        "expected_fragment": "AURAPLEX SDN. BHD.",
    },
    {
        "lang": "mixed",
        "question": "Commencement Date 是哪天？",
        "rewritten_en": "What is the commencement date?",
        "expected_source": "Letter of Appointment Internship - WU ZHONGHENG.pdf",
        "expected_fragment": "14th September 2026",
    },
]

# 失败类型补充问题（切块策略实验专用）：针对以下容易检索失败的 5 类主题，
# 每种主题补 1 个英文 + 1 个中文新表述（与 QUESTIONS 里的表述不同），共 10 题。
# 结构与 QUESTIONS 相同，但独立成列表：40 题基线（LANG_COUNTS）保持不变。
# 接地与唯一性由 test_chunking.py 自动校验。
EXTRA_QUESTIONS = [
    # ---- 实习生向谁汇报 ----
    {
        "lang": "en",
        "question": "Who does the intern report to?",
        "expected_source": "Letter of Appointment Internship - WU ZHONGHENG.pdf",
        "expected_fragment": "Khor Kai Dat",
    },
    {
        "lang": "zh",
        "question": "实习生的直属上司是谁？",
        "rewritten_en": "Who is the intern's direct supervisor?",
        "expected_source": "Letter of Appointment Internship - WU ZHONGHENG.pdf",
        "expected_fragment": "Khor Kai Dat",
    },
    # ---- 实习岗位名称 ----
    {
        "lang": "en",
        "question": "What position was the intern hired for?",
        "expected_source": "Letter of Appointment Internship - WU ZHONGHENG.pdf",
        "expected_fragment": "AI Programmer",
    },
    {
        "lang": "zh",
        "question": "实习生担任什么职位？",
        "rewritten_en": "What position does the intern hold?",
        "expected_source": "Letter of Appointment Internship - WU ZHONGHENG.pdf",
        "expected_fragment": "AI Programmer",
    },
    # ---- 需要向公司提交或回传什么文件 ----
    {
        "lang": "en",
        "question": "What letter does the university need from the company?",
        "expected_source": "university letter concerning d internship.pdf",
        "expected_fragment": "letter of confirmation",
    },
    {
        "lang": "zh",
        "question": "实习结束后实习生要归还什么？",
        "rewritten_en": "What must the intern return after the internship?",
        "expected_source": "Letter of Appointment Internship - WU ZHONGHENG.pdf",
        "expected_fragment": "return documents, equipment, and all property",
    },
    # ---- 实习期间负责开发什么 ----
    {
        "lang": "en",
        "question": "What is the intern responsible for developing?",
        "expected_source": "Letter of Appointment Internship - WU ZHONGHENG.pdf",
        "expected_fragment": "AI Server",
    },
    {
        "lang": "zh",
        "question": "实习生负责开发什么系统？",
        "rewritten_en": "What system is the intern responsible for developing?",
        "expected_source": "Letter of Appointment Internship - WU ZHONGHENG.pdf",
        "expected_fragment": "AI Server",
    },
    # ---- 公司提供什么津贴或福利 ----
    {
        "lang": "en",
        "question": "What allowance does the company provide?",
        "expected_source": "Letter of Appointment Internship - WU ZHONGHENG.pdf",
        "expected_fragment": "RM1,000.00",
    },
    {
        "lang": "zh",
        "question": "公司每个月发多少津贴？",
        "rewritten_en": "How much allowance does the company pay each month?",
        "expected_source": "Letter of Appointment Internship - WU ZHONGHENG.pdf",
        "expected_fragment": "RM1,000.00",
    },
]

LANG_COUNTS = {"en": 15, "zh": 15, "mixed": 10}
EXTRA_LANG_COUNTS = {"en": 5, "zh": 5, "mixed": 0}
