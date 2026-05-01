"""
lang/ko.py
Korean language configuration for Harness-Chandelier.

Tokenizer: kiwipiepy (LGPL v3) - commercial use allowed
Embedding: intfloat/multilingual-e5-base (MIT) by Microsoft

Note: Even with lang="ko", English text is fully supported
because multilingual-e5-base handles 100+ languages.
"""
from sklearn.feature_extraction.text import CountVectorizer


KOREAN_STOPWORDS = [
    # e5 model prefix
    "query", "passage",

    # 대명사/지시어
    "이", "그", "저", "것", "수", "이것", "그것", "저것", "이곳", "거기",
    "여기", "저기", "이쪽", "저쪽", "그쪽", "나", "너", "우리", "당신",
    "자기", "자신", "저희", "우리들", "너희", "남들", "타인", "소생", "소인",
    "제가", "저는", "저도", "나는", "나도", "나한테", "저한테",
    "이렇게", "저렇게", "그렇게",

    # 조사
    "을", "를", "은", "는", "가", "과", "와", "의", "에", "에서",
    "에게", "로", "으로", "부터", "까지", "만", "도", "로써", "으로써",
    "로서", "으로서", "뿐", "조차", "조차도", "마저", "마저도",

    # 동사/형용사 (의미 없는)
    "있다", "하다", "되다", "없다", "같다", "이다", "아니다",
    "시키다", "향하다", "도달하다", "도착하다", "오르다", "타다",
    "내리다", "넘어지다", "쓰러지다", "멈추다", "시작하다", "끝나다",
    "좋다", "나쁘다", "크다", "작다", "많다", "적다",
    "길다", "짧다", "높다", "낮다", "빠르다", "느리다", "쉽다", "어렵다",

    # 접속사/접속부사
    "그리고", "그러나", "그런데", "그래서", "그러니", "그러므로",
    "그러면", "그렇지만", "하지만", "또한", "또", "및", "혹은",
    "그래도", "그렇지", "따라서", "그리하여", "이리하여", "이어서",
    "뿐만아니라", "뿐만 아니라", "동시에", "결국", "즉", "곧",
    "그럼", "그래", "그런즉", "그러한즉", "따라", "이에",

    # 부사 (의미 약한)
    "매우", "너무", "정말", "진짜로", "조금", "약간", "많이", "거의",
    "항상", "자주", "가끔", "이미", "아직", "다시", "바로", "곧",
    "잠깐", "잠시", "여전히", "오히려", "실로", "반드시", "물론",
    "단지", "오직", "오로지", "심지어", "비록", "설마", "과연",
    "훨씬", "그냥", "그저", "혹시", "아마", "좀", "딱", "막",
    "더욱더", "비교적", "일반적으로", "구체적으로", "다소", "다수",
    "솔직히", "사실", "일단", "어차피",

    # 구어체 어미
    "해요", "어요", "아요", "네요", "죠", "군요", "구나", "하구나",
    "습니다", "습니까", "했어요", "해봐요", "해야한다", "하지마",
    "인데", "는데", "거든", "니까", "잖아", "이잖아", "이라면",
    "하면서", "하면서도", "하더라도", "할지라도", "할지언정",
    "한다면", "했다면", "됐다면",

    # 구어체 축약형 (챗봇 대화에 자주 등장)
    "근데", "근데요", "그니까", "걍", "암튼", "아무튼",
    "어쨌든", "어쨌거나", "뭐", "뭔가", "뭔지",

    # 의문사
    "어떤", "어떻게", "어떤것", "어디", "언제", "왜", "누구",
    "무엇", "얼마", "얼마나", "어느", "어찌", "어때", "어떠한",

    # 부정/조건
    "만약", "만일", "아무", "아무것", "아무도",

    # 수관형사/숫자류
    "하나", "둘", "셋", "넷", "다섯", "여섯", "일곱", "여덟", "아홉",
    "삼", "사", "오", "육", "칠", "팔", "영", "년", "월", "일",
    "첫째", "둘째", "셋째", "넷째", "다섯째",
    "여섯째", "일곱째", "여덟째", "아홉째",

    # 감탄사/의성어
    "아", "야", "어", "오", "와", "예", "네", "응", "음",
    "아이고", "아이구", "아이쿠", "아하", "오호", "허", "허허",
    "헉", "흥", "휴", "하하", "우와", "와아", "쉿",
    "세상에", "어머나", "어머", "어이구", "어이쿠", "헐",

    # 형식적 표현
    "예를 들면", "예를 들자면", "예컨대", "말하자면", "요컨대",
    "총적으로", "알 수 있다", "생각한대로", "결론을 낼 수 있다",

    # 기타 불필요
    "등", "등등", "따위", "들", "것들", "모두", "전부", "각각", "각",
    "같은", "이런", "그런", "저런", "이번", "다음", "이상", "전후",
    "날은", "분", "번", "때", "동안", "이후", "이전", "사이",
    "경우", "방법", "방식", "정도", "관련", "대한", "통해", "위해",
    "대해", "관해", "따른", "인한", "의한", "위한",
]


def _get_kiwi_tokenizer():
    """
    Returns a tokenizer function using kiwipiepy (LGPL v3).
    Extracts nouns only for clean topic keyword extraction.
    Falls back to whitespace tokenization if kiwipiepy is not installed.
    """
    try:
        from kiwipiepy import Kiwi
        kiwi = Kiwi()

        def tokenize(text):
            # 명사(N*) + 외래어(SL) 만 추출 → 영어 단어도 유지
            tokens = kiwi.tokenize(text)
            return [
                token.form for token in tokens
                if token.tag.startswith('N') or token.tag == 'SL'
            ]

        return tokenize

    except ImportError:
        # kiwipiepy 미설치 시 공백 기반 폴백
        import warnings
        warnings.warn(
            "kiwipiepy not installed. Falling back to whitespace tokenizer. "
            "Install with: pip install kiwipiepy",
            UserWarning
        )
        return lambda text: text.split()


def get_vectorizer() -> CountVectorizer:
    """
    Returns a CountVectorizer with Korean stopwords and kiwipiepy tokenizer.
    Handles Korean-English mixed text naturally.

    Requires:
        pip install kiwipiepy
    """
    return CountVectorizer(
        tokenizer=_get_kiwi_tokenizer(),
        stop_words=KOREAN_STOPWORDS
    )
