import json

# Load file count.json
with open("count.json", "r", encoding="utf-8") as f:
    count_data = json.load(f)

print("THỐNG KÊ SỐ TỪ THEO TỪNG NHÃN\n")

for label, word_dict in count_data.items():
    # Tổng số từ (tính cả lặp)
    total_words = sum(word_dict.values())

    # Số từ khác nhau
    vocab_size = len(word_dict)

    print(f"{label}")
    print(f"  - Tổng số từ (tokens): {total_words}")
    print(f"  - Số từ khác nhau (vocab): {vocab_size}\n")
