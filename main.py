import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import os
import time
from faster_whisper import WhisperModel
from openai import OpenAI
from datetime import datetime
import re

# =========================
# 1. 환경 설정 및 모델 캐싱
# =========================

OPENAI_API_KEY = "API_KEY"
client = OpenAI(api_key=OPENAI_API_KEY)

@st.cache_resource
def load_models():
    return WhisperModel("base", device="cpu", compute_type="int8")

# =========================
# 2. 핵심 처리 함수 (사용자 원본 로직 유지)
# =========================

def run_ai_inference(text):
    start_time = time.time()
    upload_date = datetime.now().strftime("%Y%m%d")
    prompt = f"""
    분석 가이드: 이 오디오는 음악일 확률이 높으나 문맥이 명백한 일반 음성(통화, 보고 등)이면 분류 리스트 중 해당하는 것으로 바꾸세요.
    [분류 리스트]: [설명문, 해명문, 보고문, 리포트, 기사문, 주장문, 평론문, 발표문, 연설문, 기안서, 품의서, 사내제안서, 기획서, 계획서, 법령문, 판결문, 공문, 거래문서, 특허명세서, 기사문, 회의록, 메모, 방송문, 보도자료, 광고문, 선전문, 논문, 연구리포트, 이력서, 계획서, 축사, 이메일, SNS, 통화, 문의 ... ]
    텍스트: "{text}"
    해당 텍스트의 분류에 알맞은 글의 주제나 목적: a
    출력규칙: 
    1. 음악 판단 시: 반드시 '추천1: [음악] 제목 - 가수', '추천2: [음악] 제목 - 가수', '추천3: [음악] 제목 - 가수' 형식을 지키세요.
    2. 음성 판단 시: 반드시 '[분류 리스트] a' 형식을 지키세요. (ex: [회의록] 학급역할배정, [문의] 환불신청, [통화] 설 연휴 약속 등)
    """
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )
    elapsed = round(time.time() - start_time, 2)
    return res.choices[0].message.content, elapsed

def parse_ai_result(ai_res):
    music_pattern = r"추천\d:\s*(.*)"
    matches = re.findall(music_pattern, ai_res)
    if matches:
        return {"type": "MUSIC", "items": [m.strip() for m in matches[:3]]}
    return {"type": "SPEECH", "items": [ai_res.strip()]}

def process_audio(f):
    total_start = time.time()
    
    t1_start = time.time()
    original_bytes = f.getvalue()
    temp_path = f"temp_{f.name}"
    with open(temp_path, "wb") as tmp:
        tmp.write(original_bytes)
    
    y, sr = librosa.load(temp_path, sr=16000)
    S_full, phase = librosa.magphase(librosa.stft(y, n_fft=1024, hop_length=512))
    S_filter = librosa.decompose.nn_filter(S_full, aggregate=np.median, metric='cosine', width=int(librosa.time_to_frames(1, sr=sr)))
    mask_vocal = librosa.util.softmask(np.maximum(0, S_full - S_filter), S_filter, power=1.5)
    y_vocal = librosa.istft(mask_vocal * S_full * phase, hop_length=512)
    
    v_path = f"vocal_{f.name}.wav"
    sf.write(v_path, y_vocal, sr, format='WAV')
    t1_elapsed = round(time.time() - t1_start, 2)
    
    #STT

    t2_start = time.time()    
    model = load_models()
    segments, _ = model.transcribe(v_path, beam_size=5)
    text = " ".join([s.text for s in segments]).strip()
    t2_elapsed = round(time.time() - t2_start, 2)
    
    raw_ai, t3_elapsed = run_ai_inference(text)
    parsed = parse_ai_result(raw_ai)
    
    if os.path.exists(temp_path): os.remove(temp_path)
    if os.path.exists(v_path): os.remove(v_path)
    
    total_elapsed = round(time.time() - total_start, 2)
    
    return {
        "original_data": original_bytes,
        "extension": f.name.split('.')[-1],
        "y_preview": y_vocal, "sr": sr, "text": text, "parsed_ai": parsed,
        "times": {
            "total": round(time.time() - total_start, 2), 
            "preprocess": t1_elapsed,
            "stt": t2_elapsed,            
            "ai": t3_elapsed,
            "total": total_elapsed            
            }
    }

# =========================
# 3. UI 컴포넌트 (Fragment)
# =========================

@st.fragment
def audio_unit_fragment(file_name):
    # 삭제 시 공간을 완전히 지우기 위한 Placeholder
    card_space = st.empty()
    
    # 세션에 데이터가 없으면 표시 안 함
    if file_name not in st.session_state.data_store:
        return

    data = st.session_state.data_store[file_name]
    
    with card_space.container(border=True):
        h_col, c_col= st.columns([0.7, 0.3])
        h_col.subheader(f"{file_name}")
        
        with c_col:
            ctrl_1, ctrl_2 = st.columns(2)
            if ctrl_1.button("다시하기", key=f"re_{file_name}"):
                
                del st.session_state.data_store[file_name]
                st.session_state.selected_titles.pop(file_name, None)
                st.empty()
                st.rerun()
            
            if ctrl_2.button("삭제", key=f"del_{file_name}"):
                # 데이터 즉시 삭제
                del st.session_state.data_store[file_name]
                st.session_state.temp_selections.pop(file_name, None)
                st.session_state.selected_titles.pop(file_name, None)
                # 시각적으로 즉시 삭제 (rerun 없이 덮어씌움)
                card_space.empty()
                return

        st.write(f"전처리: {data['times']['preprocess']}s | Whisper STT: {data['times']['stt']}s | AI 추론: {data['times']['ai']}s | 총 소요 시간: {data['times']['total']}s ")

        d_col1, d_col2 = st.columns([1, 1])
        with d_col1:
            fig, ax = plt.subplots(figsize=(10, 2.5))
            librosa.display.waveshow(data["y_preview"], sr=data["sr"], ax=ax, color='skyblue')
            st.pyplot(fig)
            plt.close(fig)
            
        with d_col2:
            st.write("**최종 제목 선택:**")
            st.info(data["text"])
            current_selected = st.session_state.selected_titles.get(file_name)
            new_selection = None

            for i, item in enumerate(data['parsed_ai']['items']):
                unique_key = f"sel_{file_name}_{i}"
                # 체크박스 상태 확인: 현재 루프의 아이템이 선택된 상태라면 체크 표시
                if st.checkbox(item, key=unique_key, value=(current_selected == item)):
                    new_selection = item

            # 루프가 종료된 후, 최종 선택된 값(new_selection)이 기존 세션값과 다르면 업데이트
            if new_selection != current_selected:
                if new_selection is None:
                    st.session_state.selected_titles.pop(file_name, None)
                else:
                    st.session_state.selected_titles[file_name] = new_selection
                st.rerun() 
                # 변경 즉시 사이드바로 정보 전달을 위해 리런
# =========================
# 4. 메인 실행부
# =========================

def main():
    st.set_page_config(layout="wide", page_title="AI Audio Pipeline")
    
    # 상태 저장소 초기화
    if "data_store" not in st.session_state: st.session_state.data_store = {}
    if "temp_selections" not in st.session_state: st.session_state.temp_selections = {}
    if "selected_titles" not in st.session_state: st.session_state.selected_titles = {}

    # 사이드바
    with st.sidebar:
        st.header("저장소")
        uploaded_files = st.file_uploader("파일 업로드", type=["wav", "mp3"], accept_multiple_files=True)
        st.divider()
        st.subheader("확정 목록")
        
        if st.session_state.selected_titles:
            for orig, final in st.session_state.selected_titles.items():
                if orig in st.session_state.data_store:
                    d = st.session_state.data_store[orig]
                    st.download_button(label=f"{final}", data=d["original_data"], 
                                     file_name=f"{final}.{d['extension']}", key=f"dl_{orig}")
        else:
            st.info("하단 확정 버튼을 눌러주세요.")

    st.title("AI 음성 분석 및 관리")

    # [핵심] 분석 루프 가드: 이미 분석된 파일은 건너뜀
    if uploaded_files:
        new_file_found = False
        for f in uploaded_files:
            if f.name not in st.session_state.data_store:
                with st.spinner(f"분석 중: {f.name}"):
                    st.session_state.data_store[f.name] = process_audio(f)
                    new_file_found = True
        if new_file_found:
            st.rerun()

    # 컨테이너 출력
    for file_name in list(st.session_state.data_store.keys()):
        audio_unit_fragment(file_name)

if __name__ == "__main__":
    main()