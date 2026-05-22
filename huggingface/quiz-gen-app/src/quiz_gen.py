from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import gradio as gr
import requests
from bs4 import BeautifulSoup
from docx import Document
from dotenv import load_dotenv
from pypdf import PdfReader


load_dotenv()

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_REFERENCE_CHARS = 6_000
MIN_REFERENCE_CHARS = 200
REQUEST_TIMEOUT_SECONDS = 20
SUPPORTED_FILE_TYPES = {".txt", ".md", ".pdf", ".docx"}
MAX_QUIZ_COUNT = 10


class ReferenceError(ValueError):
    pass


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "")
    return text.strip()


def truncate_text(text: str, limit: int = MAX_REFERENCE_CHARS) -> str:
    text = clean_text(text)
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].strip()


def extract_url_text(url: str) -> str:
    url = url.strip()
    if not url:
        return ""

    if not re.match(r"^https?://", url, flags=re.IGNORECASE):
        raise ReferenceError("URL은 http:// 또는 https://로 시작해야 합니다.")

    try:
        response = requests.get(
            url,
            timeout=REQUEST_TIMEOUT_SECONDS,
            headers={"User-Agent": "quiz-gen-app/0.1"},
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ReferenceError(f"URL을 읽을 수 없습니다: {exc}") from exc

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    main = soup.find("main") or soup.find("article") or soup.body or soup
    return clean_text(main.get_text(" "))


def extract_file_text(file: Any) -> str:
    if file is None:
        return ""

    path = Path(getattr(file, "name", file))
    suffix = path.suffix.lower()

    if suffix not in SUPPORTED_FILE_TYPES:
        allowed = ", ".join(sorted(SUPPORTED_FILE_TYPES))
        raise ReferenceError(f"지원하지 않는 파일 형식입니다. 지원 형식: {allowed}")

    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")

    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    document = Document(str(path))
    return "\n".join(paragraph.text for paragraph in document.paragraphs)


def build_reference(url: str, file: Any, text: str) -> str:
    parts = []

    url_text = extract_url_text(url) if url and url.strip() else ""
    if url_text:
        parts.append(f"[URL reference]\n{url_text}")

    file_text = extract_file_text(file)
    if file_text:
        parts.append(f"[File reference]\n{file_text}")

    if text and text.strip():
        parts.append(f"[Text reference]\n{text}")

    reference = truncate_text("\n\n".join(parts))
    if len(reference) < MIN_REFERENCE_CHARS:
        raise ReferenceError(
            f"레퍼런스가 너무 짧습니다. 최소 {MIN_REFERENCE_CHARS}자 이상의 내용을 입력하세요."
        )

    return reference


@lru_cache(maxsize=1)
def load_generator():
    from transformers import pipeline

    return pipeline(task="text-generation", model=MODEL_ID)


def build_prompt(reference: str, quiz_count: int, difficulty: str, language: str) -> str:
    quiz_count = int(quiz_count)
    language_instruction = {
        "한국어": "입력 레퍼런스가 영어, 일본어, 중국어 등 다른 언어여도 최종 퀴즈 JSON의 모든 자연어 값은 한국어로 번역해 작성하세요.",
        "English": "Even if the reference is written in another language, write every natural-language value in the final quiz JSON in English.",
    }.get(
        language,
        f"Even if the reference is written in another language, write every natural-language value in the final quiz JSON in {language}.",
    )
    sample_question = "KV 캐시가 텍스트 생성 속도를 높이는 핵심 이유는 무엇입니까?" if language == "한국어" else "Why does KV caching improve text generation speed?"
    sample_choices = (
        ["이전 Key와 Value를 재사용해 중복 계산을 줄이기 때문", "모든 입력 토큰을 삭제하기 때문", "모델의 파라미터 수를 늘리기 때문", "출력 언어를 자동으로 번역하기 때문"]
        if language == "한국어"
        else ["It reuses previous keys and values to reduce repeated computation", "It deletes all input tokens", "It increases the number of model parameters", "It automatically translates the output language"]
    )
    sample_answer = sample_choices[0]
    sample_explanation = (
        "KV 캐시는 이전 계산 결과를 저장해 새 토큰 생성 시 반복 계산을 줄입니다."
        if language == "한국어"
        else "KV caching stores previous computations so the model avoids repeating work when generating new tokens."
    )
    sample_concept = "KV 캐시" if language == "한국어" else "KV caching"
    return f"""
당신은 학습자의 개념 이해를 점검하는 교육용 퀴즈 출제자입니다.
아래 레퍼런스만 근거로 {language} 객관식 퀴즈 {quiz_count}개를 만드세요.
{language_instruction}

출제 기준:
- 난이도: {difficulty}
- 출력 언어는 반드시 "{language}" 이어야 합니다.
- 레퍼런스의 원문 언어를 따라가지 마세요. 출력 언어 선택값인 "{language}"만 따르세요.
- question은 제목이나 키워드가 아니라 학습자가 무엇을 답해야 하는지 분명한 완전한 질문 문장이어야 합니다.
- question은 반드시 물음표 또는 의문형 어미로 끝내세요.
- 단순 암기보다 핵심 개념, 원인과 결과, 비교, 적용을 물으세요.
- choices는 4개이며 서로 구분되어야 합니다.
- choices에는 실제 답변 후보 문장을 넣으세요. "보기 A", "보기 B", "선택지 1" 같은 placeholder는 절대 쓰지 마세요.
- choices에는 A/B/C/D 같은 라벨을 붙이지 말고 보기 내용만 쓰세요.
- answer는 choices 중 하나의 문자열과 정확히 같아야 합니다.
- explanation은 2문장 이내로 짧게 쓰세요.
- concept은 15자 이내의 핵심 개념명으로 쓰세요.
- 레퍼런스에 없는 사실은 만들지 마세요.

반드시 유효한 JSON만 출력하세요. Markdown 코드블록, 설명 문장, 주석은 쓰지 마세요.
최상위 JSON은 반드시 "quizzes" 키를 가진 객체여야 합니다. 배열만 출력하지 마세요.
모든 question, choices, answer, explanation, concept 값은 반드시 {language}로 작성하세요.
{{
  "quizzes": [
    {{
      "question": "{sample_question}",
      "choices": {json.dumps(sample_choices, ensure_ascii=False)},
      "answer": "{sample_answer}",
      "explanation": "{sample_explanation}",
      "concept": "{sample_concept}"
    }}
  ]
}}

레퍼런스:
{reference}

마지막 확인: 위 레퍼런스가 어떤 언어로 쓰였든 최종 JSON 내부 텍스트는 모두 "{language}"로 작성하세요.
""".strip()


def generate_text(prompt: str, quiz_count: int) -> str:
    generator = load_generator()
    messages = [
        {
            "role": "system",
            "content": "당신은 레퍼런스에 근거해 학습용 객관식 퀴즈를 만드는 교육 전문가입니다. 사용자가 선택한 출력 언어를 원문 언어보다 우선합니다.",
        },
        {"role": "user", "content": prompt},
    ]

    result = generator(
        messages,
        max_new_tokens=min(900, 220 * int(quiz_count) + 180),
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=generator.tokenizer.eos_token_id,
    )[0]["generated_text"]

    if isinstance(result, list):
        return result[-1]["content"].strip()

    return str(result).strip()


def parse_json_response(response: str) -> dict[str, Any]:
    cleaned = response.strip()
    cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned, flags=re.IGNORECASE | re.MULTILINE).strip()

    decoder = json.JSONDecoder()
    for index, char in enumerate(cleaned):
        if char not in "[{":
            continue
        try:
            parsed, _ = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            return {"quizzes": parsed}
        if isinstance(parsed, dict):
            return parsed

    quiz_objects = []
    for match in re.finditer(r"\{[^{}]*\"question\"[^{}]*\"choices\"[^{}]*\"answer\"[^{}]*\}", cleaned, re.DOTALL):
        candidate = match.group(0)
        try:
            quiz_objects.append(json.loads(candidate))
        except json.JSONDecodeError:
            continue
    if quiz_objects:
        return {"quizzes": quiz_objects}

    raise ValueError("모델 응답에서 JSON을 찾을 수 없습니다.")


def normalize_choice_text(value: Any) -> str:
    text = clean_text(str(value or "")).lower()
    text = re.sub(r"^[\(\[]?[a-d1-4가-라][\)\].:\-\s]+", "", text)
    return text.strip()


def match_answer_to_choice(answer: str, choices: list[str]) -> str:
    normalized_answer = normalize_choice_text(answer)
    if not normalized_answer:
        return answer

    labels = {
        "a": 0,
        "1": 0,
        "가": 0,
        "b": 1,
        "2": 1,
        "나": 1,
        "c": 2,
        "3": 2,
        "다": 2,
        "d": 3,
        "4": 3,
        "라": 3,
    }
    compact_answer = re.sub(r"[\s\W_]+", "", str(answer or "").lower())
    if compact_answer in labels and labels[compact_answer] < len(choices):
        return choices[labels[compact_answer]]

    for choice in choices:
        if normalize_choice_text(choice) == normalized_answer:
            return choice

    for choice in choices:
        normalized_choice = normalize_choice_text(choice)
        if normalized_answer in normalized_choice or normalized_choice in normalized_answer:
            return choice

    return answer


def normalize_quizzes(data: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(data, list):
        quizzes = data
    else:
        quizzes = data.get("quizzes")

    if not isinstance(quizzes, list):
        return []

    normalized = []
    for quiz in quizzes[:MAX_QUIZ_COUNT]:
        if not isinstance(quiz, dict):
            continue

        question = quiz.get("question", "").strip()
        choices = [
            re.sub(r"^[\(\[]?[A-Da-d1-4가-라][\)\].:\-\s]+", "", str(choice).strip())
            for choice in quiz.get("choices") or []
            if str(choice).strip()
        ]
        answer = quiz.get("answer", "").strip()
        explanation = quiz.get("explanation", "").strip()
        concept = quiz.get("concept", "").strip()
        matched_answer = match_answer_to_choice(answer, choices)

        placeholder_pattern = re.compile(r"^(보기|선택지|choice|option)\s*[a-d1-4가-라]?$", re.IGNORECASE)
        has_placeholder = any(placeholder_pattern.match(normalize_choice_text(choice)) for choice in choices)

        if question and len(choices) >= 2 and matched_answer and not has_placeholder:
            normalized.append(
                {
                    "question": question,
                    "choices": choices[:4],
                    "answer": matched_answer,
                    "explanation": explanation,
                    "concept": concept,
                }
            )

    return normalized


def create_quiz(
    input_mode: str,
    url: str,
    file: Any,
    text: str,
    quiz_count: int,
    difficulty: str,
    language: str,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> list[Any]:
    try:
        if input_mode != "직접 입력":
            raise ReferenceError("현재는 직접 입력만 사용할 수 있습니다. 파일 업로드와 URL 입력은 준비 중입니다.")

        progress(0.05, desc="레퍼런스 읽는 중")
        selected_url = url if input_mode == "URL" else ""
        selected_file = file if input_mode == "파일 업로드" else None
        selected_text = text if input_mode == "직접 입력" else ""
        reference = build_reference(selected_url, selected_file, selected_text)
        progress(0.25, desc="퀴즈 생성 프롬프트 준비 중")
        prompt = build_prompt(reference, quiz_count, difficulty, language)
        progress(0.35, desc=f"{MODEL_ID} 모델로 퀴즈 생성 중")
        response = generate_text(prompt, quiz_count)
        progress(0.85, desc="생성 결과 파싱 중")
        parsed = parse_json_response(response)
        quizzes = normalize_quizzes(parsed)
        if not quizzes:
            raise ValueError(f"생성된 퀴즈를 읽을 수 없습니다.\n\n{response}")

        progress(0.95, desc="퀴즈 화면 구성 중")
        status = f"{len(quizzes)}개 문항이 생성되었습니다. 답을 선택한 뒤 제출하세요."
        return [
            quizzes,
            gr.update(value=status, visible=True),
        ]
    except ReferenceError as exc:
        return [
            [],
            gr.update(value=f"입력 오류: {exc}", visible=True),
        ]
    except Exception as exc:
        return [
            [],
            gr.update(value=f"퀴즈 생성 중 오류가 발생했습니다: {exc}", visible=True),
        ]


def grade_quiz(quizzes: list[dict[str, Any]], *answers: str) -> str:
    if not quizzes:
        return "채점할 퀴즈가 없습니다. 먼저 퀴즈를 생성하세요."

    correct_count = 0
    blocks = []
    for index, quiz in enumerate(quizzes):
        selected = answers[index] if index < len(answers) else None
        answer = quiz["answer"]
        is_correct = normalize_choice_text(selected) == normalize_choice_text(answer)
        correct_count += int(is_correct)
        result = "정답" if is_correct else "오답"
        selected_text = selected or "선택 안 함"

        blocks.append(
            f"### {index + 1}. {result}\n\n"
            f"**문제:** {quiz['question']}\n\n"
            f"**내 답:** {selected_text}\n\n"
            f"**정답:** {answer}\n\n"
            f"**해설:** {quiz.get('explanation', '')}\n\n"
            f"**개념:** {quiz.get('concept', '')}"
        )

    score = round((correct_count / len(quizzes)) * 100)
    return f"## 점수: {correct_count}/{len(quizzes)} ({score}점)\n\n" + "\n\n---\n\n".join(blocks)


def grade_single_quiz(quizzes: list[dict[str, Any]], quiz_index: int, selected: str) -> str:
    if not quizzes or quiz_index >= len(quizzes):
        return "채점할 문제가 없습니다."

    quiz = quizzes[quiz_index]
    answer = quiz["answer"]
    is_correct = normalize_choice_text(selected) == normalize_choice_text(answer)
    result = "정답" if is_correct else "오답"
    selected_text = selected or "선택 안 함"

    return (
        f"### {result}\n\n"
        f"**내 답:** {selected_text}\n\n"
        f"**정답:** {answer}\n\n"
        f"**해설:** {quiz.get('explanation', '')}\n\n"
        f"**개념:** {quiz.get('concept', '')}"
    )


def start_generation() -> list[Any]:
    return [
        [],
        gr.update(value="퀴즈를 생성하는 중입니다. 잠시 기다려 주세요.", visible=True),
    ]


def switch_input_mode(input_mode: str) -> list[Any]:
    return [
        gr.update(visible=input_mode == "직접 입력"),
        gr.update(visible=input_mode == "파일 업로드"),
        gr.update(visible=input_mode == "URL"),
    ]


with gr.Blocks(title="Quiz Generator") as app:
    quiz_state = gr.State([])
    gr.Markdown("# Quiz Generator")
    gr.Markdown("URL, 파일, 텍스트 레퍼런스를 바탕으로 개념 이해를 확인하는 객관식 퀴즈를 생성합니다.")

    with gr.Row():
        with gr.Column(scale=1):
            input_mode = gr.Radio(
                label="입력 방식",
                choices=["직접 입력", "파일 업로드", "URL"],
                value="직접 입력",
            )
            with gr.Group(visible=True) as text_group:
                text_input = gr.Textbox(label="직접 입력", lines=10, placeholder="학습 레퍼런스를 붙여넣으세요.")
            with gr.Group(visible=False) as file_group:
                gr.Markdown("파일 업로드는 준비 중입니다.")
                file_input = gr.File(label="파일 업로드", file_types=list(SUPPORTED_FILE_TYPES), interactive=False)
            with gr.Group(visible=False) as url_group:
                gr.Markdown("URL 입력은 준비 중입니다.")
                url_input = gr.Textbox(
                    label="URL",
                    placeholder="https://example.com/reference",
                    interactive=False,
                )

            input_mode.change(
                fn=switch_input_mode,
                inputs=input_mode,
                outputs=[text_group, file_group, url_group],
            )

            with gr.Row():
                count_input = gr.Slider(
                    label="문항 수",
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                )
                difficulty_input = gr.Dropdown(
                    label="난이도",
                    choices=["쉬움", "보통", "어려움"],
                    value="보통",
                )

            language_input = gr.Dropdown(
                label="출력 언어",
                choices=["한국어", "English"],
                value="한국어",
            )
            generate_button = gr.Button("퀴즈 생성", variant="primary")

        with gr.Column(scale=1):
            status_output = gr.Markdown(visible=False)

            @gr.render(inputs=quiz_state)
            def render_quizzes(quizzes: list[dict[str, Any]]) -> None:
                if not quizzes:
                    return

                for index, quiz in enumerate(quizzes):
                    with gr.Group():
                        gr.Markdown(f"### {index + 1}. {quiz['question']}")
                        answer_input = gr.Radio(
                            label="답 선택",
                            choices=quiz["choices"],
                            value=None,
                        )
                        submit_answer = gr.Button("제출", variant="primary")
                        feedback = gr.Markdown(visible=False)

                        submit_answer.click(
                            fn=lambda selected, current_quizzes, quiz_index=index: grade_single_quiz(
                                current_quizzes,
                                quiz_index,
                                selected,
                            ),
                            inputs=[answer_input, quiz_state],
                            outputs=feedback,
                        ).then(
                            fn=lambda: gr.update(visible=True),
                            outputs=feedback,
                        )

    generate_button.click(
        fn=start_generation,
        outputs=[
            quiz_state,
            status_output,
        ],
        show_progress="hidden",
    ).then(
        fn=create_quiz,
        inputs=[
            input_mode,
            url_input,
            file_input,
            text_input,
            count_input,
            difficulty_input,
            language_input,
        ],
        outputs=[
            quiz_state,
            status_output,
        ],
        show_progress="full",
    )


if __name__ == "__main__":
    app.launch(debug=True)
