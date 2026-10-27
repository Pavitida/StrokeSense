# StrokeSense — Frontend Handoff

## 1. Project Overview
StrokeSense is an experimental multimodal AI prototype using face images, speech audio, and movement videos.

## 2. Frontend Responsibilities
- Build the Streamlit user interface.
- Accept three uploaded files: image, audio, video.
- Save uploaded files temporarily.
- Call the backend pipeline.
- Display results and errors.
- Delete temporary files after processing.

## 3. Backend Entry Point
File: src/fusion/run_pipeline.py

Function: run_pipeline(image_path, audio_path, video_path)

All three arguments must be local file paths.

## 4. Example Backend Call

    from src.fusion.run_pipeline import run_pipeline

    result = run_pipeline(
        image_path="face.jpg",
        audio_path="speech.wav",
        video_path="movement.mp4"
    )

## 5. Successful Response
- status: success
- face: face model result
- speech: speech model result
- movement: movement model result
- fusion: experimental combined score and method

Read the individual scores using:
- result["face"]["score"]
- result["speech"]["score"]
- result["movement"]["score"]
- result["fusion"]["score"]

## 6. Incomplete Response
If status is incomplete:
- failed_modality identifies the failed module.
- reason explains the failure.
- fusion is None.

Always check status before reading fusion.

## 7. Setup
From the project root:

    python -m pip install -r requirements.txt
    python -m pip install streamlit

## 8. Run Streamlit
After creating app.py:

    python -m streamlit run app.py

## 9. Important Limitations
- The fusion score is an equal-weight average.
- It is not a validated stroke probability.
- The movement model uses rehabilitation exercise data.
- StrokeSense is not a medical diagnosis tool.

## 10. Team Responsibilities
Backend team: AI inference, models, pipeline.
Frontend team: Streamlit UI, file uploads, result display.

Do not modify the AI models without coordinating with the backend team.
