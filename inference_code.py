"""
==============================================================================
PHẦN 4: INFERENCE & BLEU EVALUATION
==============================================================================
Tuân thủ yêu cầu đồ án:
- Hàm translate(sentence: str) -> str
- Đánh giá BLEU bằng nltk.translate.bleu_score
- Demo dịch 5 câu từ tập Test
==============================================================================
"""

import torch
import random
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm

# ==============================================================================
# 1. HÀM TRANSLATE (ĐÚNG SIGNATURE YÊU CẦU)
# ==============================================================================
def translate(sentence: str) -> str:
    """
    Dịch một câu tiếng Anh sang tiếng Pháp.
    
    Args:
        sentence: Câu tiếng Anh cần dịch (string)
        
    Returns:
        Câu tiếng Pháp đã dịch (string)
        
    Lưu ý: Hàm này sử dụng các biến global:
        - model, device, vocab_en, vocab_fr, tokenizer_en
        - SOS_IDX, EOS_IDX
    """
    MAX_LEN = 50
    
    model.eval()
    
    # ===== 1. TOKENIZE =====
    # Dùng spacy tokenizer
    tokens = tokenizer_en(sentence.lower())
    
    # Thêm <sos> và <eos>
    tokens = ['<sos>'] + tokens + ['<eos>']
    
    # ===== 2. NUMERICALIZE =====
    # Chuyển token thành index
    src_indexes = [vocab_en[token] for token in tokens]
    
    # ===== 3. TENSORIZE =====
    # Shape: [seq_len, 1] (batch_size = 1)
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)
    
    # ===== 4. TÍNH SRC_LEN =====
    # ⚠️ QUAN TRỌNG: src_len PHẢI nằm trên CPU
    src_len = torch.tensor([len(src_indexes)], dtype=torch.long)  # CPU
    
    # ===== 5. ENCODER FORWARD =====
    with torch.no_grad():
        hidden, cell = model.encoder(src_tensor, src_len)
    
    # ===== 6. GREEDY DECODING =====
    # Bắt đầu với <sos>
    trg_indexes = [SOS_IDX]
    
    for _ in range(MAX_LEN):
        # Lấy token cuối làm input
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
        
        with torch.no_grad():
            output, hidden, cell = model.decoder(trg_tensor, hidden, cell)
        
        # Greedy: chọn từ có xác suất cao nhất (argmax)
        pred_token = output.argmax(1).item()
        
        trg_indexes.append(pred_token)
        
        # Dừng khi gặp <eos>
        if pred_token == EOS_IDX:
            break
    
    # ===== 7. CONVERT TO WORDS =====
    # Chuyển index thành từ vựng
    trg_tokens = [vocab_fr.get_itos()[i] for i in trg_indexes]
    
    # Bỏ <sos> đầu và <eos> cuối (nếu có)
    if trg_tokens[0] == '<sos>':
        trg_tokens = trg_tokens[1:]
    if '<eos>' in trg_tokens:
        trg_tokens = trg_tokens[:trg_tokens.index('<eos>')]
    
    # Trả về câu dịch dạng string
    return ' '.join(trg_tokens)


# ==============================================================================
# 2. HÀM TÍNH BLEU SCORE (Dùng NLTK)
# ==============================================================================
def calculate_bleu_score(test_src, test_trg, num_samples=None):
    """
    Tính điểm BLEU trung bình trên tập Test.
    
    Args:
        test_src: List câu nguồn (tiếng Anh)
        test_trg: List câu đích (tiếng Pháp)
        num_samples: Số mẫu để đánh giá (None = toàn bộ)
        
    Returns:
        bleu_avg: Điểm BLEU trung bình (0-1)
    """
    # Smoothing function để xử lý trường hợp n-gram = 0
    smooth = SmoothingFunction().method1
    
    total_bleu = 0
    count = 0
    
    # Giới hạn số mẫu nếu cần
    if num_samples:
        indices = random.sample(range(len(test_src)), min(num_samples, len(test_src)))
    else:
        indices = range(len(test_src))
    
    print("Đang tính BLEU Score...")
    
    for idx in tqdm(indices, desc="Calculating BLEU"):
        src_sentence = test_src[idx]
        trg_sentence = test_trg[idx]
        
        # Dịch câu
        pred_sentence = translate(src_sentence)
        
        # Tokenize prediction và reference
        pred_tokens = pred_sentence.split()
        ref_tokens = tokenizer_fr(trg_sentence.lower())
        
        # NLTK sentence_bleu yêu cầu:
        # - references: list of list of tokens (có thể nhiều reference)
        # - hypothesis: list of tokens
        reference = [ref_tokens]  # Wrap trong list
        hypothesis = pred_tokens
        
        # Tính BLEU cho câu này
        try:
            bleu = sentence_bleu(reference, hypothesis, smoothing_function=smooth)
            total_bleu += bleu
            count += 1
        except:
            # Bỏ qua nếu có lỗi
            continue
    
    bleu_avg = total_bleu / count if count > 0 else 0
    return bleu_avg


# ==============================================================================
# 3. HÀM DEMO DỊCH
# ==============================================================================
def demo_translation(test_src, test_trg, num_examples=5):
    """
    Demo dịch một số câu ngẫu nhiên từ tập Test.
    
    Args:
        test_src: List câu nguồn
        test_trg: List câu đích
        num_examples: Số câu ví dụ
    """
    print("\n" + "=" * 70)
    print(" DEMO DỊCH MẪU")
    print("=" * 70)
    
    # Chọn ngẫu nhiên
    indices = random.sample(range(len(test_src)), num_examples)
    
    for i, idx in enumerate(indices, 1):
        src = test_src[idx]
        trg = test_trg[idx]
        pred = translate(src)
        
        print(f"\n--- Ví dụ {i} ---")
        print(f"📥 Source (EN):     {src}")
        print(f"📌 Reference (FR):  {trg}")
        print(f"🤖 Predicted (FR):  {pred}")
    
    print("\n" + "=" * 70)


# ==============================================================================
# 4. MAIN EXECUTION
# ==============================================================================

print("=" * 70)
print(" PHẦN 4: INFERENCE & BLEU EVALUATION")
print("=" * 70)

# ----- LOAD BEST MODEL -----
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()
print("✅ Đã load model từ 'best_model.pth'\n")

# ----- DEMO: DỊCH 5 CÂU NGẪU NHIÊN TỪ TẬP TEST -----
demo_translation(test_en, test_fr, num_examples=5)

# ----- TÍNH BLEU SCORE -----
print("\n" + "=" * 70)
print(" ĐÁNH GIÁ BLEU SCORE TRÊN TẬP TEST")
print("=" * 70)

# Tính BLEU trên toàn bộ tập test (hoặc giới hạn để chạy nhanh)
bleu_score_avg = calculate_bleu_score(test_en, test_fr, num_samples=None)

print("\n" + "=" * 70)
print(f" 🎯 BLEU SCORE TRUNG BÌNH: {bleu_score_avg * 100:.2f}")
print("=" * 70)

# ----- BẢNG ĐÁNH GIÁ BLEU -----
print("""
📊 HƯỚNG DẪN ĐÁNH GIÁ BLEU SCORE:
┌─────────────────┬──────────────────────────────────┐
│ BLEU Score      │ Đánh giá                         │
├─────────────────┼──────────────────────────────────┤
│ < 10%           │ Kém - Gần như vô nghĩa           │
│ 10% - 19%       │ Yếu - Khó hiểu                   │
│ 20% - 29%       │ Trung bình - Hiểu ý chính        │
│ 30% - 40%       │ Khá - Chất lượng chấp nhận được  │
│ 40% - 50%       │ Tốt - Chất lượng cao             │
│ > 50%           │ Rất tốt - Gần với người dịch     │
└─────────────────┴──────────────────────────────────┘

📌 Lưu ý: Mô hình Seq2Seq cơ bản (không Attention) thường đạt 15-25%.
""")


# ==============================================================================
# 5. DỊCH THỬ CÂU TỰ NHẬP
# ==============================================================================
print("\n" + "=" * 70)
print(" DỊCH THỬ CÂU TÙY Ý")
print("=" * 70)

test_sentences = [
    "I love machine learning.",
    "The weather is beautiful today.",
    "A man is walking with his dog.",
]

for sentence in test_sentences:
    result = translate(sentence)
    print(f"EN: {sentence}")
    print(f"FR: {result}")
    print("-" * 50)


# ==============================================================================
# 6. CHẾ ĐỘ TƯƠNG TÁC (Optional - Uncomment để sử dụng)
# ==============================================================================
"""
def interactive_mode():
    print("\\n" + "=" * 70)
    print(" CHẾ ĐỘ DỊCH TƯƠNG TÁC")
    print(" Nhập 'quit' để thoát")
    print("=" * 70)
    
    while True:
        sentence = input("\\n📝 Nhập câu tiếng Anh: ").strip()
        
        if sentence.lower() == 'quit':
            print("👋 Tạm biệt!")
            break
        
        if not sentence:
            print("⚠️ Vui lòng nhập câu!")
            continue
        
        result = translate(sentence)
        print(f"🇫🇷 Kết quả: {result}")

# Uncomment để chạy:
# interactive_mode()
"""
