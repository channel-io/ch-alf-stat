def detect_language(text: str) -> str:
    detections = []
    for ch in text:
        if '\uac00' <= ch <= '\ud7a3' or '\u1100' <= ch <= '\u11ff':
            detections.append("ko")
        elif ('\u3040' <= ch <= '\u309f') or ('\u30a0' <= ch <= '\u30ff') or ('\u4e00' <= ch <= '\u9fff'):
            detections.append("ja")
        elif ch.isalpha():
            detections.append("en")

    if len(detections) == 0:
        return "ko"
    
    korean_ratio = detections.count("ko") / len(detections)
    japanese_ratio = detections.count("ja") / len(detections)

    if korean_ratio > 0.5:
        return "ko"
    elif japanese_ratio > 0.5:
        return "ja"
    else:
        return "en"