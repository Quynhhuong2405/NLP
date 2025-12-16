#!/usr/bin/env python
# coding: utf-8

# # 📦 PHẦN 1: TẢI VÀ XỬ LÝ DỮ LIỆU
# 
# ---
# 
# **Mục tiêu:**
# 1. Tải dataset Multi30K (English → French)
# 2. Tokenization với SpaCy
# 3. Xây dựng Vocabulary với special tokens: `<unk>`, `<pad>`, `<sos>`, `<eos>`
# 4. Tạo DataLoader với sorting + padding (sẵn sàng cho LSTM)
# 
# ---

# In[ ]:


# ==============================================================================
# CELL 1.1: CÀI ĐẶT THƯ VIỆN & CẤU HÌNH
# ==============================================================================

# Cài đặt thư viện (chạy trên Google Colab)
# 'pip install torch==2.2.2 torchtext==0.17.2 -q')
# 'pip install spacy nltk -q')
# 'python -m spacy download en_core_web_sm -q')
# 'python -m spacy download fr_core_news_sm -q')

# Import thư viện
import torch
import torch.nn as nn
import torch.nn.functional as F  # Cần cho Attention
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import io
import os
import random

# =============================================================================
# SEED cho Reproducibility
# =============================================================================
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"✅ Torch version: {torch.__version__}")
print(f"✅ Torchtext version: {torchtext.__version__}")
print(f"✅ Device: {device}")
print(f"✅ Seed: {SEED}")

# =============================================================================
# TẢI DATASET MULTI30K (EN-FR)
# =============================================================================
# 'mkdir -p data')
# 'wget -q https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/train.en.gz -O data/train.en.gz')
# 'wget -q https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/train.fr.gz -O data/train.fr.gz')
# 'wget -q https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/val.en.gz -O data/val.en.gz')
# 'wget -q https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/val.fr.gz -O data/val.fr.gz')
# 'wget -q https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/test_2016_flickr.en.gz -O data/test.en.gz')
# 'wget -q https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/test_2016_flickr.fr.gz -O data/test.fr.gz')

# Giải nén
# 'gunzip -kf data/*.gz')
# 'ls -la data/')

print("\n✅ Đã chuẩn bị xong dữ liệu và thư viện!")


# In[23]:


# ==============================================================================
# CELL 1.2: ĐỌC VÀ KIỂM TRA DỮ LIỆU
# ==============================================================================

def read_data(en_file, fr_file):
    """
    Đọc dữ liệu song ngữ từ file.
    
    Returns:
        en_sentences: List câu tiếng Anh
        fr_sentences: List câu tiếng Pháp (căn chỉnh 1-1)
    """
    with open(en_file, 'r', encoding='utf-8') as f:
        en_sentences = [line.strip() for line in f.readlines() if line.strip()]
    with open(fr_file, 'r', encoding='utf-8') as f:
        fr_sentences = [line.strip() for line in f.readlines() if line.strip()]
    
    # Đảm bảo số câu khớp nhau
    assert len(en_sentences) == len(fr_sentences), "Số câu EN và FR không khớp!"
    return en_sentences, fr_sentences

# Đọc dữ liệu train, val, test từ folder 'data/'
train_en, train_fr = read_data('data/train.en', 'data/train.fr')
val_en, val_fr = read_data('data/val.en', 'data/val.fr')
test_en, test_fr = read_data('data/test.en', 'data/test.fr')

print("=" * 50)
print("📊 THỐNG KÊ DỮ LIỆU MULTI30K")
print("=" * 50)
print(f"   Train:      {len(train_en):,} cặp câu")
print(f"   Validation: {len(val_en):,} cặp câu")
print(f"   Test:       {len(test_en):,} cặp câu")
print("=" * 50)

# Hiển thị ví dụ
print("\n📝 VÍ DỤ 5 CẶP CÂU ĐẦU TIÊN:")
for i in range(5):
    print(f"\nExample {i+1}:")
    print(f"   EN: {train_en[i]}")
    print(f"   FR: {train_fr[i]}")


# In[24]:


# ==============================================================================
# CELL 1.3: TOKENIZATION & VOCABULARY
# ==============================================================================

# --- CẤU HÌNH TOKEN ĐẶC BIỆT ---
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
SPECIAL_SYMBOLS = ['<unk>', '<pad>', '<sos>', '<eos>']

# =============================================================================
# HÀM: get_tokenizers()
# Nhiệm vụ: Tải tokenizer của SpaCy cho tiếng Anh và Pháp
# =============================================================================
def get_tokenizers():
    try:
        en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
        fr_tokenizer = get_tokenizer('spacy', language='fr_core_news_sm')
        print("✅ Đã tải thành công Tokenizer (SpaCy)")
        return en_tokenizer, fr_tokenizer
    except OSError:
        print("❌ Lỗi: Chưa tìm thấy model SpaCy. Hãy chạy lại Cell 1.1")
        return None, None

# =============================================================================
# HÀM: build_vocab()
# Nhiệm vụ: Xây dựng vocabulary từ file dữ liệu
# =============================================================================
def build_vocab(filepath, tokenizer):
    """
    Xây dựng vocabulary từ file text.
    
    Args:
        filepath: Đường dẫn file text
        tokenizer: Tokenizer function
    
    Returns:
        vocab: Vocabulary object với special tokens
    """
    def yield_tokens(path):
        with io.open(path, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    yield tokenizer(line.strip())
    
    print(f"   Đang xây dựng vocab từ {filepath}...")
    
    vocab = build_vocab_from_iterator(
        yield_tokens(filepath),
        min_freq=2,           # Bỏ qua từ xuất hiện < 2 lần
        max_tokens=10000,     # Giới hạn vocabulary size
        specials=SPECIAL_SYMBOLS
    )
    
    # Set default index cho từ không có trong vocab (OOV)
    vocab.set_default_index(UNK_IDX)
    return vocab

# =============================================================================
# THỰC THI
# =============================================================================
tokenizer_en, tokenizer_fr = get_tokenizers()

if tokenizer_en and tokenizer_fr:
    print("\n📚 Xây dựng Vocabulary:")
    vocab_en = build_vocab('data/train.en', tokenizer_en)
    vocab_fr = build_vocab('data/train.fr', tokenizer_fr)
    
    print("\n" + "=" * 50)
    print("📊 BÁO CÁO VOCABULARY")
    print("=" * 50)
    print(f"   Vocab EN: {len(vocab_en):,} tokens (max: 10,004)")
    print(f"   Vocab FR: {len(vocab_fr):,} tokens (max: 10,004)")
    print(f"   Special tokens: <unk>={UNK_IDX}, <pad>={PAD_IDX}, <sos>={SOS_IDX}, <eos>={EOS_IDX}")
    
    # Kiểm tra xử lý từ lạ (OOV)
    test_oov = vocab_en['từ_không_tồn_tại_xyz']
    if test_oov == UNK_IDX:
        print("   OOV handling: ✅ Thành công (trả về index 0)")
    else:
        print(f"   OOV handling: ❌ Thất bại (trả về {test_oov})")
    print("=" * 50)


# In[25]:


# ==============================================================================
# CELL 1.4: TEXT TRANSFORM (Numericalization)
# ==============================================================================

def text_transform(text, tokenizer, vocab):
    """
    Chuyển đổi câu text thành tensor số.
    
    Pipeline: text → tokens → token_ids → tensor với <sos> và <eos>
    
    Args:
        text: Câu input (string)
        tokenizer: Tokenizer function
        vocab: Vocabulary object
    
    Returns:
        tensor: [SOS, token_ids..., EOS]
    """
    tokens = tokenizer(text)
    token_ids = [vocab[token] for token in tokens]
    return torch.tensor([SOS_IDX] + token_ids + [EOS_IDX], dtype=torch.long)

# =============================================================================
# KIỂM TRA TEXT PIPELINE
# =============================================================================
print("=" * 50)
print("🔍 KIỂM TRA TEXT PIPELINE")
print("=" * 50)

sample_sentence = "Two young, White males are outside."
print(f"1. Câu gốc:     '{sample_sentence}'")

# Tokenize
sample_tokens = tokenizer_en(sample_sentence)
print(f"2. Tokens:      {sample_tokens}")

# Transform
sample_tensor = text_transform(sample_sentence, tokenizer_en, vocab_en)
print(f"3. Tensor:      {sample_tensor}")
print(f"4. Shape:       {sample_tensor.shape}")
print(f"5. Dtype:       {sample_tensor.dtype}")

# Kiểm tra logic <sos> và <eos>
if sample_tensor[0] == SOS_IDX and sample_tensor[-1] == EOS_IDX:
    print("\n✅ Logic <sos>/<eos>: ĐÚNG (đầu=2, cuối=3)")
else:
    print(f"\n❌ Logic <sos>/<eos>: SAI (đầu={sample_tensor[0]}, cuối={sample_tensor[-1]})")
print("=" * 50)


# In[26]:


# ==============================================================================
# CELL 1.5: DATASET, COLLATE_FN & DATALOADER
# ==============================================================================

# =============================================================================
# CLASS: TranslationDataset
# =============================================================================
class TranslationDataset(Dataset):
    """
    Dataset cho dữ liệu song ngữ.
    Lưu trữ cặp câu (source, target) dạng text.
    """
    def __init__(self, src_list, trg_list):
        self.src_list = src_list
        self.trg_list = trg_list
        assert len(src_list) == len(trg_list), "Số câu source và target không khớp!"

    def __len__(self):
        return len(self.src_list)

    def __getitem__(self, idx):
        return self.src_list[idx], self.trg_list[idx]

# =============================================================================
# HÀM: collate_fn
# Xử lý batch: padding + sorting theo độ dài (cho pack_padded_sequence)
# =============================================================================
def collate_fn(batch):
    """
    Xử lý batch dữ liệu:
    1. Chuyển text → tensor
    2. Padding để đồng bộ độ dài
    3. Sort theo độ dài giảm dần (yêu cầu cho pack_padded_sequence)
    
    Returns:
        src_padded: [src_len, batch_size]
        trg_padded: [trg_len, batch_size]
        sorted_lens: [batch_size] - độ dài thực của từng câu source
    """
    src_batch, trg_batch = [], []

    # Chuyển đổi Text → Tensor
    for src_sample, trg_sample in batch:
        src_batch.append(text_transform(src_sample, tokenizer_en, vocab_en))
        trg_batch.append(text_transform(trg_sample, tokenizer_fr, vocab_fr))

    # Padding (đồng bộ độ dài trong batch)
    src_padded = pad_sequence(src_batch, padding_value=PAD_IDX)
    trg_padded = pad_sequence(trg_batch, padding_value=PAD_IDX)

    # Tính độ dài thực tế của từng câu nguồn (dtype=long cho pack_padded_sequence)
    src_lens = torch.tensor([len(x) for x in src_batch], dtype=torch.long)

    # Sort giảm dần theo độ dài (yêu cầu của pack_padded_sequence với enforce_sorted=True)
    sorted_lens, sorted_indices = torch.sort(src_lens, descending=True)

    # Sắp xếp lại tensors theo thứ tự đã sort
    src_padded = src_padded[:, sorted_indices]
    trg_padded = trg_padded[:, sorted_indices]

    return src_padded, trg_padded, sorted_lens

# =============================================================================
# TẠO DATALOADER
# =============================================================================
BATCH_SIZE = 64

# Tạo Dataset
train_dataset = TranslationDataset(train_en, train_fr)
valid_dataset = TranslationDataset(val_en, val_fr)
test_dataset = TranslationDataset(test_en, test_fr)

# Tạo DataLoader
train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True,           # Shuffle cho training
    collate_fn=collate_fn
)
valid_loader = DataLoader(
    valid_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False,
    collate_fn=collate_fn
)
test_loader = DataLoader(
    test_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False,
    collate_fn=collate_fn
)

# =============================================================================
# KIỂM TRA DATALOADER
# =============================================================================
print("=" * 50)
print("🔍 KIỂM TRA DATALOADER")
print("=" * 50)

try:
    src, trg, src_len = next(iter(train_loader))
    
    print(f"1. Source shape:   {src.shape} (seq_len, batch_size)")
    print(f"2. Target shape:   {trg.shape} (seq_len, batch_size)")
    print(f"3. Lengths shape:  {src_len.shape}")
    print(f"   - Câu dài nhất:  {src_len[0]} tokens")
    print(f"   - Câu ngắn nhất: {src_len[-1]} tokens")
    
    # Kiểm tra sorting
    if src_len[0] >= src_len[-1]:
        print("\n✅ Batch đã SORT theo độ dài (giảm dần)")
        print("   → Sẵn sàng cho pack_padded_sequence!")
    else:
        print("\n❌ Lỗi: Batch chưa được sort đúng!")
    
    print("\n" + "-" * 50)
    print(f"📊 SỐ BATCH:")
    print(f"   Train:      {len(train_loader)} batches")
    print(f"   Validation: {len(valid_loader)} batches")
    print(f"   Test:       {len(test_loader)} batches")
    print("=" * 50)
    print("\n✅ PHẦN 1 HOÀN TẤT - Sẵn sàng cho PHẦN 2 (Model)!")

except Exception as e:
    print(f"\n❌ LỖI: {e}")
    print("Gợi ý: Kiểm tra lại collate_fn có trả về đúng 3 giá trị không?")


# # 🏗️ PHẦN 2: XÂY DỰNG MÔ HÌNH BASELINE SEQ2SEQ
# 
# ---
# 
# ## Kiến trúc Encoder-Decoder LSTM (Không Attention)
# 
# ```
# ┌─────────────────────────────────────────────────────────────────┐
# │                     BASELINE SEQ2SEQ ARCHITECTURE                │
# ├─────────────────────────────────────────────────────────────────┤
# │                                                                  │
# │  INPUT (EN)       ENCODER           DECODER        OUTPUT (FR)  │
# │  ─────────→      ────────→         ────────→      ──────────→   │
# │                                                                  │
# │  "A man walks" → [LSTM x2] → (h,c) → [LSTM x2] → "Un homme..." │
# │                              ↑                                   │
# │                       Context Vector                             │
# │                  (Fixed representation)                          │
# │                                                                  │
# └─────────────────────────────────────────────────────────────────┘
# ```
# 
# **Công thức:**
# - Encoder: `(h_t, c_t) = LSTM(embed(x_t), (h_{t-1}, c_{t-1}))`
# - Decoder: `(h_t, c_t) = LSTM(embed(y_{t-1}), (h'_{t-1}, c'_{t-1}))`
# - Output:  `p(y_t) = softmax(Linear(h_t))`
# 
# **Hyperparameters:**
# 
# | Parameter | Value | Mô tả |
# |-----------|-------|-------|
# | `EMB_DIM` | 256 | Embedding dimension |
# | `HID_DIM` | 512 | Hidden state dimension |
# | `N_LAYERS` | 2 | Số lớp LSTM |
# | `DROPOUT` | 0.5 | Dropout rate |
# 
# ---

# In[27]:


# ==============================================================================
# CELL 2.1: ĐỊNH NGHĨA CÁC CLASS MODEL
# ==============================================================================

# =============================================================================
# CLASS: Encoder
# =============================================================================
class Encoder(nn.Module):
    """
    Encoder LSTM cho Seq2Seq.
    
    Nhiệm vụ: Đọc câu nguồn và tạo context vector (hidden, cell states).
    Sử dụng pack_padded_sequence để xử lý padding hiệu quả.
    
    Args:
        input_dim: Kích thước vocabulary nguồn
        emb_dim: Embedding dimension
        hid_dim: Hidden state dimension
        n_layers: Số lớp LSTM
        dropout: Dropout rate
    """
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len):
        """
        Forward pass của Encoder.
        
        Args:
            src: [src_len, batch_size] - Tensor câu nguồn
            src_len: [batch_size] - Độ dài thực của mỗi câu
        
        Returns:
            hidden: [n_layers, batch_size, hid_dim] - Hidden states
            cell: [n_layers, batch_size, hid_dim] - Cell states
        
        Note: Baseline KHÔNG trả về encoder_outputs (sẽ thêm khi có Attention)
        """
        # src: [src_len, batch_size]
        embedded = self.dropout(self.embedding(src))
        # embedded: [src_len, batch_size, emb_dim]
        
        # Pack để LSTM không xử lý padding tokens
        packed_embedded = pack_padded_sequence(embedded, src_len.cpu(), enforce_sorted=True)
        
        packed_outputs, (hidden, cell) = self.lstm(packed_embedded)
        # hidden: [n_layers, batch_size, hid_dim]
        # cell: [n_layers, batch_size, hid_dim]
        
        return hidden, cell


# =============================================================================
# CLASS: Decoder
# =============================================================================
class Decoder(nn.Module):
    """
    Decoder LSTM cho Seq2Seq (Baseline - không Attention).
    
    Nhiệm vụ: Sinh câu đích từng token một, dựa trên context từ Encoder.
    
    Args:
        output_dim: Kích thước vocabulary đích
        emb_dim: Embedding dimension
        hid_dim: Hidden state dimension
        n_layers: Số lớp LSTM
        dropout: Dropout rate
    """
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        """
        Forward pass của Decoder (1 timestep).
        
        Args:
            input: [batch_size] - Token hiện tại
            hidden: [n_layers, batch_size, hid_dim] - Hidden states
            cell: [n_layers, batch_size, hid_dim] - Cell states
        
        Returns:
            prediction: [batch_size, output_dim] - Logits cho vocabulary
            hidden: [n_layers, batch_size, hid_dim] - Updated hidden
            cell: [n_layers, batch_size, hid_dim] - Updated cell
        """
        # input: [batch_size] → [1, batch_size]
        input = input.unsqueeze(0)
        
        embedded = self.dropout(self.embedding(input))
        # embedded: [1, batch_size, emb_dim]
        
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        # output: [1, batch_size, hid_dim]
        
        prediction = self.fc_out(output.squeeze(0))
        # prediction: [batch_size, output_dim]
        
        return prediction, hidden, cell


# =============================================================================
# CLASS: Seq2Seq
# =============================================================================
class Seq2Seq(nn.Module):
    """
    Mô hình Seq2Seq Baseline (Encoder-Decoder không Attention).
    
    Điều phối Encoder và Decoder, xử lý teacher forcing.
    
    Args:
        encoder: Encoder instance
        decoder: Decoder instance
        device: 'cuda' hoặc 'cpu'
    """
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        # Đảm bảo encoder và decoder tương thích
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must match!"
        assert encoder.n_layers == decoder.n_layers, \
            "Number of layers of encoder and decoder must match!"

    def forward(self, src, src_len, trg, teacher_forcing_ratio=0.5):
        """
        Forward pass của Seq2Seq.
        
        Args:
            src: [src_len, batch_size] - Câu nguồn
            src_len: [batch_size] - Độ dài câu nguồn
            trg: [trg_len, batch_size] - Câu đích
            teacher_forcing_ratio: Tỷ lệ sử dụng ground truth (0.0 - 1.0)
        
        Returns:
            outputs: [trg_len, batch_size, output_dim] - Logits cho mỗi timestep
        """
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        # Tensor lưu output của decoder tại mỗi timestep
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        # Encode câu nguồn
        hidden, cell = self.encoder(src, src_len)
        
        # Token đầu tiên là <sos>
        input = trg[0, :]
        
        # Decode từng timestep
        for t in range(1, trg_len):
            # Forward decoder
            output, hidden, cell = self.decoder(input, hidden, cell)
            
            # Lưu output
            outputs[t] = output
            
            # Teacher Forcing: dùng ground truth hoặc prediction
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1
        
        return outputs


# =============================================================================
# HÀM: init_weights
# =============================================================================
def init_weights(m):
    """
    Khởi tạo weights cho model.
    - Weights: Uniform distribution [-0.08, 0.08]
    - Biases: 0
    """
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.uniform_(param.data, -0.08, 0.08)
        else:
            nn.init.constant_(param.data, 0)


print("✅ Đã định nghĩa: Encoder, Decoder, Seq2Seq, init_weights")


# In[28]:


# ==============================================================================
# CELL 2.2: KHỞI TẠO BASELINE MODEL (FIXED CONTEXT VECTOR)
# ==============================================================================
# ⚠️ ĐÂY LÀ MÔ HÌNH BASELINE - SỬ DỤNG CONTEXT VECTOR CỐ ĐỊNH
# (hidden state cuối cùng của Encoder, KHÔNG có Attention)

# =============================================================================
# HYPERPARAMETERS
# =============================================================================
INPUT_DIM = len(vocab_en)    # Vocab size tiếng Anh
OUTPUT_DIM = len(vocab_fr)   # Vocab size tiếng Pháp
ENC_EMB_DIM = 256            # Encoder embedding dim
DEC_EMB_DIM = 256            # Decoder embedding dim
HID_DIM = 512                # Hidden state dim
N_LAYERS = 2                 # Số lớp LSTM
ENC_DROPOUT = 0.5            # Encoder dropout
DEC_DROPOUT = 0.5            # Decoder dropout

# =============================================================================
# KHỞI TẠO BASELINE MODEL
# =============================================================================
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

# ⚠️ QUAN TRỌNG: Đặt tên biến là baseline_model để phân biệt với attention_model
baseline_model = Seq2Seq(enc, dec, device).to(device)

# Áp dụng weight initialization
baseline_model.apply(init_weights)

# Đếm parameters
baseline_params = sum(p.numel() for p in baseline_model.parameters() if p.requires_grad)

print("=" * 60)
print("🏗️ MÔ HÌNH BASELINE SEQ2SEQ (FIXED CONTEXT VECTOR)")
print("=" * 60)
print("📌 Đặc điểm: Sử dụng CONTEXT VECTOR CỐ ĐỊNH")
print("   (Chỉ dùng hidden state cuối cùng của Encoder)")
print("-" * 60)
print(f"Device:           {device}")
print(f"Input dim (EN):   {INPUT_DIM:,}")
print(f"Output dim (FR):  {OUTPUT_DIM:,}")
print(f"Embedding dim:    {ENC_EMB_DIM}")
print(f"Hidden dim:       {HID_DIM}")
print(f"Num layers:       {N_LAYERS}")
print(f"Dropout:          {ENC_DROPOUT}")
print("-" * 60)
print(f"Total parameters: {baseline_params:,}")
print("=" * 60)

# =============================================================================
# KIỂM TRA KẾT NỐI DATA → MODEL
# =============================================================================
print("\n" + "=" * 60)
print("🔍 KIỂM TRA FORWARD PASS (BASELINE)")
print("=" * 60)

try:
    # Lấy 1 batch từ train_loader
    src, trg, src_len = next(iter(train_loader))
    src = src.to(device)
    trg = trg.to(device)
    
    print(f"Input (src):      {src.shape}  [seq_len, batch_size]")
    print(f"Input (src_len):  {src_len.shape}  [batch_size]")
    print(f"Target (trg):     {trg.shape}  [seq_len, batch_size]")
    
    # Forward pass với baseline_model
    output = baseline_model(src, src_len, trg)
    
    print(f"Output:           {output.shape}  [seq_len, batch_size, vocab_size]")
    
    # Validate output shape
    expected_shape = (trg.shape[0], trg.shape[1], OUTPUT_DIM)
    if output.shape == expected_shape:
        print("\n✅ BASELINE FORWARD PASS THÀNH CÔNG!")
        print("   → Baseline Model sẵn sàng cho Training.")
    else:
        print(f"\n⚠️ Shape không khớp! Expected: {expected_shape}")
    
except Exception as e:
    print(f"\n❌ LỖI: {e}")
    import traceback
    traceback.print_exc()

print("=" * 60)
print("\n✅ PHẦN 2 HOÀN TẤT - Baseline Model đã sẵn sàng!")


# # 🎯 PHẦN 3: SEQ2SEQ + LUONG ATTENTION (MÔ HÌNH CHÍNH)
# 
# ---
# 
# ## Kiến trúc Encoder-Decoder LSTM với Attention
# 
# ```
# ┌─────────────────────────────────────────────────────────────────────────┐
# │                    SEQ2SEQ + LUONG ATTENTION ARCHITECTURE               │
# ├─────────────────────────────────────────────────────────────────────────┤
# │                                                                         │
# │  ENCODER                              DECODER                           │
# │  ────────                              ────────                          │
# │                                                                         │
# │  "A man walks"  ──────┐                                                 │
# │       ↓               │                                                 │
# │   [LSTM x2]           │              ┌─────────────────┐               │
# │       ↓               │              │   ATTENTION     │               │
# │  encoder_outputs ─────┼──────────────│  (Luong General)│               │
# │  (h₁, h₂, ..., hₙ)    │              │                 │               │
# │       ↓               │              │  score = hₜᵀWₐhₛ│               │
# │  (hidden, cell) ──────┼──────────────│  α = softmax    │               │
# │                       │              │  c = Σ αᵢhᵢ     │               │
# │                       │              └────────┬────────┘               │
# │                       │                       ↓                         │
# │                       │              context_vector                     │
# │                       │                       ↓                         │
# │                       └──────────────→ [embed; context]                 │
# │                                              ↓                          │
# │                                         [LSTM x2]                       │
# │                                              ↓                          │
# │                                    [hidden; context]                    │
# │                                              ↓                          │
# │                                         Linear → vocab                  │
# │                                              ↓                          │
# │                                      "Un homme marche"                  │
# │                                                                         │
# └─────────────────────────────────────────────────────────────────────────┘
# ```
# 
# ## Luong Attention (General)
# 
# **Công thức:**
# 
# $$score(h_t, h_s) = h_t^T \cdot W_a \cdot h_s$$
# 
# $$\alpha = softmax(score)$$
# 
# $$context = \sum_i \alpha_i \cdot h_i$$
# 
# **Tham khảo:** Luong et al. (2015) - "Effective Approaches to Attention-based NMT"
# 
# ---

# In[1]:


# ==============================================================================
# CELL 3.1: ATTENTION MECHANISM + ENCODER/DECODER VỚI ATTENTION
# ==============================================================================

# =============================================================================
# CLASS: Attention (Luong General)
# =============================================================================
class Attention(nn.Module):
    """
    Luong General Attention.
    
    Công thức: score(h_t, h_s) = h_t^T * W_a * h_s
    
    Args:
        hid_dim: Hidden dimension của encoder/decoder
    
    Reference: "Effective Approaches to Attention-based NMT" (Luong et al., 2015)
    """
    def __init__(self, hid_dim):
        super().__init__()
        # W_a: Linear layer để tính score
        self.W_a = nn.Linear(hid_dim, hid_dim, bias=False)
    
    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        """
        Tính attention weights và context vector.
        
        Args:
            decoder_hidden: [batch_size, hid_dim] - Hidden state hiện tại của decoder
            encoder_outputs: [src_len, batch_size, hid_dim] - Tất cả hidden states của encoder
            mask: [batch_size, src_len] - Mask cho padding (1=valid, 0=pad)
        
        Returns:
            context: [batch_size, hid_dim] - Context vector
            attention_weights: [batch_size, src_len] - Attention weights
        """
        # decoder_hidden: [batch_size, hid_dim]
        # encoder_outputs: [src_len, batch_size, hid_dim]
        
        src_len = encoder_outputs.shape[0]
        
        # Reshape decoder_hidden: [batch_size, hid_dim] → [batch_size, 1, hid_dim]
        decoder_hidden = decoder_hidden.unsqueeze(1)
        
        # Permute encoder_outputs: [src_len, batch_size, hid_dim] → [batch_size, src_len, hid_dim]
        encoder_outputs_perm = encoder_outputs.permute(1, 0, 2)
        
        # Tính W_a * h_s: [batch_size, src_len, hid_dim]
        energy = self.W_a(encoder_outputs_perm)
        
        # Tính h_t^T * (W_a * h_s): [batch_size, 1, hid_dim] x [batch_size, hid_dim, src_len]
        # → [batch_size, 1, src_len]
        attention_scores = torch.bmm(decoder_hidden, energy.permute(0, 2, 1))
        
        # Squeeze: [batch_size, 1, src_len] → [batch_size, src_len]
        attention_scores = attention_scores.squeeze(1)
        
        # Áp dụng mask (nếu có) - đặt padding positions = -inf
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e10)
        
        # Softmax để có attention weights: [batch_size, src_len]
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Tính context vector: weighted sum của encoder outputs
        # [batch_size, 1, src_len] x [batch_size, src_len, hid_dim] → [batch_size, 1, hid_dim]
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs_perm)
        
        # Squeeze: [batch_size, 1, hid_dim] → [batch_size, hid_dim]
        context = context.squeeze(1)
        
        return context, attention_weights


# =============================================================================
# CLASS: EncoderAttention (Sửa từ Encoder để trả về encoder_outputs)
# =============================================================================
class EncoderAttention(nn.Module):
    """
    Encoder LSTM cho Seq2Seq + Attention.
    
    Khác với Encoder baseline: Trả về thêm encoder_outputs cho Attention.
    
    Args:
        input_dim: Kích thước vocabulary nguồn
        emb_dim: Embedding dimension
        hid_dim: Hidden state dimension
        n_layers: Số lớp LSTM
        dropout: Dropout rate
    """
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len):
        """
        Forward pass của Encoder (cho Attention).
        
        Args:
            src: [src_len, batch_size] - Tensor câu nguồn
            src_len: [batch_size] - Độ dài thực của mỗi câu
        
        Returns:
            encoder_outputs: [src_len, batch_size, hid_dim] - Tất cả hidden states
            hidden: [n_layers, batch_size, hid_dim] - Hidden state cuối
            cell: [n_layers, batch_size, hid_dim] - Cell state cuối
        """
        # src: [src_len, batch_size]
        embedded = self.dropout(self.embedding(src))
        # embedded: [src_len, batch_size, emb_dim]
        
        # Pack để LSTM không xử lý padding
        packed_embedded = pack_padded_sequence(embedded, src_len.cpu(), enforce_sorted=True)
        
        packed_outputs, (hidden, cell) = self.lstm(packed_embedded)
        
        # Unpack để lấy encoder_outputs cho Attention
        encoder_outputs, _ = pad_packed_sequence(packed_outputs)
        # encoder_outputs: [src_len, batch_size, hid_dim]
        # hidden: [n_layers, batch_size, hid_dim]
        # cell: [n_layers, batch_size, hid_dim]
        
        return encoder_outputs, hidden, cell


# =============================================================================
# CLASS: DecoderAttention (Decoder với Luong Attention)
# =============================================================================
class DecoderAttention(nn.Module):
    """
    Decoder LSTM với Luong Attention.
    
    Kiến trúc:
    1. Embed input token
    2. Tính attention weights và context vector
    3. Concat [embedding; context] → LSTM input
    4. LSTM forward
    5. Concat [hidden; context] → Linear → vocab
    
    Args:
        output_dim: Kích thước vocabulary đích
        emb_dim: Embedding dimension
        hid_dim: Hidden state dimension
        n_layers: Số lớp LSTM
        dropout: Dropout rate
    """
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.attention = Attention(hid_dim)
        
        # LSTM nhận: embedding + context = emb_dim + hid_dim
        self.lstm = nn.LSTM(emb_dim + hid_dim, hid_dim, n_layers, dropout=dropout)
        
        # Linear nhận: hidden + context = hid_dim * 2
        self.fc_out = nn.Linear(hid_dim * 2, output_dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, encoder_outputs, mask):
        """
        Forward pass của Decoder với Attention (1 timestep).
        
        Args:
            input: [batch_size] - Token hiện tại
            hidden: [n_layers, batch_size, hid_dim] - Hidden states
            cell: [n_layers, batch_size, hid_dim] - Cell states
            encoder_outputs: [src_len, batch_size, hid_dim] - Encoder outputs
            mask: [batch_size, src_len] - Mask cho padding
        
        Returns:
            prediction: [batch_size, output_dim] - Logits cho vocabulary
            hidden: [n_layers, batch_size, hid_dim] - Updated hidden
            cell: [n_layers, batch_size, hid_dim] - Updated cell
            attention_weights: [batch_size, src_len] - Attention weights
        """
        # input: [batch_size] → [1, batch_size]
        input = input.unsqueeze(0)
        
        # Embed
        embedded = self.dropout(self.embedding(input))
        # embedded: [1, batch_size, emb_dim]
        
        # Tính Attention: dùng hidden state của layer cuối cùng
        # hidden[-1]: [batch_size, hid_dim]
        context, attention_weights = self.attention(hidden[-1], encoder_outputs, mask)
        # context: [batch_size, hid_dim]
        # attention_weights: [batch_size, src_len]
        
        # Concat embedding và context cho LSTM input
        # embedded: [1, batch_size, emb_dim]
        # context: [batch_size, hid_dim] → [1, batch_size, hid_dim]
        lstm_input = torch.cat([embedded, context.unsqueeze(0)], dim=2)
        # lstm_input: [1, batch_size, emb_dim + hid_dim]
        
        # LSTM forward
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        # output: [1, batch_size, hid_dim]
        
        # Concat hidden và context cho prediction
        # output.squeeze(0): [batch_size, hid_dim]
        # context: [batch_size, hid_dim]
        combined = torch.cat([output.squeeze(0), context], dim=1)
        # combined: [batch_size, hid_dim * 2]
        
        # Prediction
        prediction = self.fc_out(combined)
        # prediction: [batch_size, output_dim]
        
        return prediction, hidden, cell, attention_weights


# =============================================================================
# CLASS: Seq2SeqAttention
# =============================================================================
class Seq2SeqAttention(nn.Module):
    """
    Mô hình Seq2Seq với Luong Attention (Mô hình chính).
    
    Điều phối EncoderAttention và DecoderAttention.
    
    Args:
        encoder: EncoderAttention instance
        decoder: DecoderAttention instance
        device: 'cuda' hoặc 'cpu'
        pad_idx: Index của padding token
    """
    def __init__(self, encoder, decoder, device, pad_idx):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.pad_idx = pad_idx
        
        # Đảm bảo encoder và decoder tương thích
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must match!"
        assert encoder.n_layers == decoder.n_layers, \
            "Number of layers of encoder and decoder must match!"

    def create_mask(self, src):
        """
        Tạo mask cho padding positions.
        
        Args:
            src: [src_len, batch_size]
        
        Returns:
            mask: [batch_size, src_len] - 1 cho valid, 0 cho padding
        """
        mask = (src != self.pad_idx).permute(1, 0)
        # mask: [batch_size, src_len]
        return mask

    def forward(self, src, src_len, trg, teacher_forcing_ratio=0.5):
        """
        Forward pass của Seq2Seq với Attention.
        
        Args:
            src: [src_len, batch_size] - Câu nguồn
            src_len: [batch_size] - Độ dài câu nguồn
            trg: [trg_len, batch_size] - Câu đích
            teacher_forcing_ratio: Tỷ lệ sử dụng ground truth
        
        Returns:
            outputs: [trg_len, batch_size, output_dim] - Logits cho mỗi timestep
        """
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        # Tensor lưu outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        # Encode - giờ trả về thêm encoder_outputs
        encoder_outputs, hidden, cell = self.encoder(src, src_len)
        # encoder_outputs: [src_len, batch_size, hid_dim]
        # hidden: [n_layers, batch_size, hid_dim]
        # cell: [n_layers, batch_size, hid_dim]
        
        # Tạo mask cho padding
        mask = self.create_mask(src)
        # mask: [batch_size, src_len]
        
        # Token đầu tiên là <sos>
        input = trg[0, :]
        
        # Decode từng timestep
        for t in range(1, trg_len):
            # Forward decoder với attention
            output, hidden, cell, attention_weights = self.decoder(
                input, hidden, cell, encoder_outputs, mask
            )
            # output: [batch_size, output_dim]
            # attention_weights: [batch_size, src_len]
            
            # Lưu output
            outputs[t] = output
            
            # Teacher Forcing
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1
        
        return outputs


print("✅ Đã định nghĩa: Attention, EncoderAttention, DecoderAttention, Seq2SeqAttention")


# In[ ]:


# ==============================================================================
# CELL 3.2: KHỞI TẠO ATTENTION MODEL VÀ KIỂM TRA
# ==============================================================================

# =============================================================================
# KHỞI TẠO MODEL VỚI ATTENTION
# =============================================================================
# Hyperparameters giữ nguyên từ baseline
enc_attn = EncoderAttention(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec_attn = DecoderAttention(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

# ⚠️ QUAN TRỌNG: Đặt tên biến là attention_model để phân biệt với baseline_model
attention_model = Seq2SeqAttention(enc_attn, dec_attn, device, PAD_IDX).to(device)

# Áp dụng weight initialization
attention_model.apply(init_weights)

# Đếm parameters
attention_params = sum(p.numel() for p in attention_model.parameters() if p.requires_grad)

print("=" * 60)
print("🎯 MÔ HÌNH SEQ2SEQ + LUONG ATTENTION")
print("=" * 60)
print("📌 Đặc điểm: Sử dụng DYNAMIC CONTEXT VECTOR")
print("   (Attention weights thay đổi theo từng timestep)")
print("-" * 60)
print(f"Device:           {device}")
print(f"Input dim (EN):   {INPUT_DIM:,}")
print(f"Output dim (FR):  {OUTPUT_DIM:,}")
print(f"Embedding dim:    {ENC_EMB_DIM}")
print(f"Hidden dim:       {HID_DIM}")
print(f"Num layers:       {N_LAYERS}")
print(f"Dropout:          {ENC_DROPOUT}")
print("-" * 60)
print(f"Attention params: {attention_params:,}")
print(f"Baseline params:  {baseline_params:,}")
print(f"Thêm (Attention): {attention_params - baseline_params:,}")
print("=" * 60)

# =============================================================================
# KIỂM TRA FORWARD PASS
# =============================================================================
print("\n" + "=" * 60)
print("🔍 KIỂM TRA FORWARD PASS (ATTENTION MODEL)")
print("=" * 60)

try:
    # Lấy 1 batch từ train_loader
    src, trg, src_len = next(iter(train_loader))
    src = src.to(device)
    trg = trg.to(device)
    
    print(f"Input (src):      {src.shape}  [seq_len, batch_size]")
    print(f"Input (src_len):  {src_len.shape}  [batch_size]")
    print(f"Target (trg):     {trg.shape}  [seq_len, batch_size]")
    
    # Test Encoder
    encoder_outputs, hidden, cell = attention_model.encoder(src, src_len)
    print(f"\nEncoder outputs:  {encoder_outputs.shape}  [src_len, batch_size, hid_dim]")
    print(f"Hidden:           {hidden.shape}  [n_layers, batch_size, hid_dim]")
    print(f"Cell:             {cell.shape}  [n_layers, batch_size, hid_dim]")
    
    # Test full forward pass
    output = attention_model(src, src_len, trg)
    print(f"\nOutput:           {output.shape}  [trg_len, batch_size, vocab_size]")
    
    # Validate output shape
    expected_shape = (trg.shape[0], trg.shape[1], OUTPUT_DIM)
    if output.shape == expected_shape:
        print("\n✅ ATTENTION FORWARD PASS THÀNH CÔNG!")
        print("   → Attention Model sẵn sàng cho Training.")
    else:
        print(f"\n⚠️ Shape không khớp! Expected: {expected_shape}")
    
except Exception as e:
    print(f"\n❌ LỖI: {e}")
    import traceback
    traceback.print_exc()

print("=" * 60)

# ⚠️ KHÔNG gán lại biến - sử dụng baseline_model và attention_model riêng biệt
# Điều này đảm bảo cả hai mô hình tồn tại độc lập để so sánh

print("\n" + "=" * 60)
print("📌 HAI MÔ HÌNH ĐÃ KHỞI TẠO:")
print("   1. baseline_model  - Seq2Seq với Fixed Context Vector")
print("   2. attention_model - Seq2Seq với Luong Attention")
print("=" * 60)
print("\n✅ PHẦN 3 HOÀN TẤT - Sẵn sàng cho PHẦN 4 (Training)!")


# # 🚀 PHẦN 4: TRAINING PROCESS
# 
# ---
# 
# ## Quy trình huấn luyện
# 
# ```
# ┌─────────────────────────────────────────────────────────────────────┐
# │                         TRAINING LOOP                               │
# ├─────────────────────────────────────────────────────────────────────┤
# │                                                                     │
# │  for epoch in range(N_EPOCHS):                                      │
# │      ├── train_loss = train(model, train_loader, ...)              │
# │      │       ├── Forward pass (với Teacher Forcing 0.5)            │
# │      │       ├── Tính loss (bỏ <sos>, ignore <pad>)                │
# │      │       ├── Backward + Gradient Clipping                       │
# │      │       └── Update weights                                     │
# │      │                                                              │
# │      ├── valid_loss = evaluate(model, valid_loader, ...)           │
# │      │       └── Forward pass (không Teacher Forcing)              │
# │      │                                                              │
# │      ├── if valid_loss < best_loss:                                │
# │      │       └── Save checkpoint (best_model.pth)                  │
# │      │                                                              │
# │      └── if no_improvement >= PATIENCE:                            │
# │              └── EARLY STOPPING                                     │
# │                                                                     │
# └─────────────────────────────────────────────────────────────────────┘
# ```
# 
# **Hyperparameters (theo ĐỒ ÁN):**
# 
# | Parameter | Value | Mô tả |
# |-----------|-------|-------|
# | `N_EPOCHS` | 20 | Số epoch tối đa |
# | `LEARNING_RATE` | 0.001 | Learning rate (Adam) |
# | `CLIP` | 1.0 | Gradient clipping |
# | `TEACHER_FORCING` | 0.5 | Tỷ lệ Teacher Forcing |
# | `PATIENCE` | 3 | Early Stopping patience |
# 
# ---

# In[ ]:


# ==============================================================================
# CELL 4.1: CẤU HÌNH TRAINING VÀ HELPER FUNCTIONS
# ==============================================================================

import time
import math

# =============================================================================
# HYPERPARAMETERS (THEO ĐỒ ÁN)
# =============================================================================
N_EPOCHS = 20                    # Số epoch tối đa
CLIP = 1.0                       # Gradient clipping
LEARNING_RATE = 0.001            # Learning rate
PATIENCE = 3                     # Early Stopping patience
TEACHER_FORCING_RATIO = 0.5      # Tỷ lệ Teacher Forcing

# =============================================================================
# HELPER FUNCTION
# =============================================================================
def epoch_time(start_time, end_time):
    """Tính thời gian chạy 1 epoch (phút, giây)."""
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

print("✅ Đã cấu hình hyperparameters:")
print(f"   N_EPOCHS = {N_EPOCHS}")
print(f"   LEARNING_RATE = {LEARNING_RATE}")
print(f"   CLIP = {CLIP}")
print(f"   TEACHER_FORCING_RATIO = {TEACHER_FORCING_RATIO}")
print(f"   PATIENCE = {PATIENCE}")


# In[ ]:


# ==============================================================================
# CELL 4.2: HÀM TRAIN VÀ EVALUATE
# ==============================================================================

def train(model, iterator, optimizer, criterion, clip, device, teacher_forcing_ratio=0.5):
    """
    Huấn luyện model trong 1 epoch.
    
    Args:
        model: Mô hình Seq2Seq (với Attention)
        iterator: DataLoader train
        optimizer: Adam optimizer
        criterion: CrossEntropyLoss (với ignore_index=PAD_IDX)
        clip: Gradient clipping value
        device: 'cuda' hoặc 'cpu'
        teacher_forcing_ratio: Tỷ lệ sử dụng Teacher Forcing
        
    Returns:
        epoch_loss: Loss trung bình của epoch
    """
    model.train()
    epoch_loss = 0
    
    progress_bar = tqdm(iterator, desc="Training", leave=False)
    
    for batch in progress_bar:
        # ===== 1. UNPACK BATCH =====
        src, trg, src_len = batch
        
        # Chuyển src, trg lên device
        src = src.to(device)         # [src_len, batch_size]
        trg = trg.to(device)         # [trg_len, batch_size]
        # ⚠️ src_len PHẢI nằm trên CPU cho pack_padded_sequence!
        
        # ===== 2. FORWARD PASS =====
        optimizer.zero_grad()
        
        # Forward với teacher_forcing_ratio
        output = model(src, src_len, trg, teacher_forcing_ratio)
        # output: [trg_len, batch_size, output_dim]
        
        # ===== 3. TÍNH LOSS =====
        # 📌 LOGIC SLICING:
        # - output[0] là zeros tensor (do loop bắt đầu từ t=1)
        # - trg[0] là <sos> token
        # - Phải bỏ cả hai trước khi tính loss
        output_dim = output.shape[-1]
        
        output = output[1:]   # [trg_len-1, batch_size, output_dim]
        trg = trg[1:]         # [trg_len-1, batch_size]
        
        # Reshape về 2D cho CrossEntropyLoss
        output = output.reshape(-1, output_dim)  # [(trg_len-1)*batch_size, output_dim]
        trg = trg.reshape(-1)                    # [(trg_len-1)*batch_size]
        
        loss = criterion(output, trg)
        
        # ===== 4. BACKWARD PASS =====
        loss.backward()
        
        # Gradient clipping để tránh exploding gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion, device):
    """
    Đánh giá model trên tập validation/test.
    
    Args:
        model: Mô hình Seq2Seq
        iterator: DataLoader val/test
        criterion: CrossEntropyLoss
        device: 'cuda' hoặc 'cpu'
        
    Returns:
        epoch_loss: Loss trung bình
    """
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(iterator, desc="Evaluating", leave=False):
            src, trg, src_len = batch
            
            src = src.to(device)
            trg = trg.to(device)
            # src_len giữ nguyên trên CPU
            
            # Forward với teacher_forcing_ratio = 0 (không dùng ground truth)
            output = model(src, src_len, trg, teacher_forcing_ratio=0)
            
            output_dim = output.shape[-1]
            
            output = output[1:]
            trg = trg[1:]
            
            output = output.reshape(-1, output_dim)
            trg = trg.reshape(-1)
            
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)


print("✅ Đã định nghĩa: train(), evaluate()")


# ## 4.1 HUẤN LUYỆN BASELINE SEQ2SEQ (FIXED CONTEXT VECTOR)
# 
# ---
# 
# **Mục tiêu:** Huấn luyện mô hình Baseline để chứng minh hoạt động của context vector cố định.
# 
# > ⚠️ **QUAN TRỌNG:** Đây là bước BẮT BUỘC để đáp ứng tiêu chí 1 (3.0 điểm).
# 
# ---

# In[ ]:


# ==============================================================================
# CELL 4.1a: HUẤN LUYỆN BASELINE SEQ2SEQ (FIXED CONTEXT VECTOR)
# ==============================================================================

from tqdm.auto import tqdm
import time
import math

print("=" * 60)
print("🏗️ HUẤN LUYỆN BASELINE SEQ2SEQ (FIXED CONTEXT VECTOR)")
print("=" * 60)
print("📌 Mô hình này sử dụng context vector CỐ ĐỊNH")
print("   (Chỉ hidden state cuối cùng của Encoder)")
print("=" * 60 + "\n")

# =============================================================================
# CẤU HÌNH TRAINING CHO BASELINE
# =============================================================================
BASELINE_EPOCHS = 3              # Đủ để chứng minh mô hình hoạt động
LEARNING_RATE = 0.001
CLIP = 1.0
TEACHER_FORCING_RATIO = 0.5

# Optimizer & Criterion cho Baseline
baseline_optimizer = torch.optim.Adam(baseline_model.parameters(), lr=LEARNING_RATE)
baseline_criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

# History tracking
baseline_history = {
    'train_loss': [],
    'valid_loss': [],
    'train_ppl': [],
    'valid_ppl': []
}

print(f"Epochs:           {BASELINE_EPOCHS}")
print(f"Learning Rate:    {LEARNING_RATE}")
print(f"Gradient Clip:    {CLIP}")
print(f"Teacher Forcing:  {TEACHER_FORCING_RATIO}")
print("=" * 60 + "\n")

# =============================================================================
# TRAINING LOOP CHO BASELINE
# =============================================================================
best_baseline_loss = float('inf')

for epoch in range(BASELINE_EPOCHS):
    start_time = time.time()
    
    # ===== TRAIN =====
    baseline_model.train()
    train_loss = 0
    
    for batch in tqdm(train_loader, desc=f"Baseline Epoch {epoch+1}/{BASELINE_EPOCHS}", leave=False):
        src, trg, src_len = batch
        src = src.to(device)
        trg = trg.to(device)
        
        baseline_optimizer.zero_grad()
        
        # Forward pass
        output = baseline_model(src, src_len, trg, TEACHER_FORCING_RATIO)
        
        # Reshape for loss
        output_dim = output.shape[-1]
        output = output[1:].reshape(-1, output_dim)
        trg = trg[1:].reshape(-1)
        
        # Calculate loss
        loss = baseline_criterion(output, trg)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(baseline_model.parameters(), CLIP)
        baseline_optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    
    # ===== EVALUATE =====
    baseline_model.eval()
    valid_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(valid_loader, desc="Validating", leave=False):
            src, trg, src_len = batch
            src = src.to(device)
            trg = trg.to(device)
            
            output = baseline_model(src, src_len, trg, 0)  # No teacher forcing
            
            output_dim = output.shape[-1]
            output = output[1:].reshape(-1, output_dim)
            trg = trg[1:].reshape(-1)
            
            loss = baseline_criterion(output, trg)
            valid_loss += loss.item()
    
    valid_loss /= len(valid_loader)
    
    # Calculate perplexity
    train_ppl = math.exp(train_loss)
    valid_ppl = math.exp(valid_loss)
    
    # Save history
    baseline_history['train_loss'].append(train_loss)
    baseline_history['valid_loss'].append(valid_loss)
    baseline_history['train_ppl'].append(train_ppl)
    baseline_history['valid_ppl'].append(valid_ppl)
    
    # Save best model
    if valid_loss < best_baseline_loss:
        best_baseline_loss = valid_loss
        torch.save(baseline_model.state_dict(), 'baseline_model.pth')
        save_status = "✅ Model saved!"
    else:
        save_status = ""
    
    end_time = time.time()
    epoch_mins = int((end_time - start_time) / 60)
    epoch_secs = int((end_time - start_time) % 60)
    
    print(f"Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s")
    print(f"\tTrain Loss: {train_loss:.3f} | Train PPL: {train_ppl:7.3f}")
    print(f"\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {valid_ppl:7.3f} {save_status}")
    print("-" * 60)

print("\n" + "=" * 60)
print("🎉 BASELINE TRAINING HOÀN TẤT!")
print("=" * 60)
print(f"Best Validation Loss: {best_baseline_loss:.3f}")
print(f"Best Validation PPL:  {math.exp(best_baseline_loss):.3f}")
print(f"Model đã lưu tại:     'baseline_model.pth'")
print("=" * 60)


# ## 4.2 HUẤN LUYỆN ATTENTION SEQ2SEQ (MÔ HÌNH CHÍNH)
# 
# ---
# 
# **Mục tiêu:** Huấn luyện mô hình Seq2Seq + Luong Attention với Early Stopping.
# 
# **So sánh với Baseline:**
# - Baseline: Context vector CỐ ĐỊNH (chỉ hidden state cuối)
# - Attention: Context vector ĐỘNG (weighted sum của tất cả encoder outputs)
# 
# ---

# In[ ]:


# ==============================================================================
# CELL 4.3: KHỞI TẠO OPTIMIZER & CRITERION CHO ATTENTION MODEL
# ==============================================================================

# Optimizer: Adam cho attention_model
attention_optimizer = torch.optim.Adam(attention_model.parameters(), lr=LEARNING_RATE)

# Loss function: CrossEntropyLoss với ignore_index để bỏ qua PAD token
attention_criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

# Cấu hình training
N_EPOCHS = 20                    # Số epoch tối đa
PATIENCE = 3                     # Early Stopping patience

print("=" * 60)
print("🎯 CẤU HÌNH HUẤN LUYỆN ATTENTION MODEL")
print("=" * 60)
print(f"Device:              {device}")
print(f"Model:               {attention_model.__class__.__name__}")
print(f"Total Parameters:    {attention_params:,}")
print("-" * 60)
print(f"Optimizer:           Adam (lr={LEARNING_RATE})")
print(f"Loss:                CrossEntropyLoss (ignore_index={PAD_IDX})")
print(f"Epochs:              {N_EPOCHS} (max)")
print(f"Gradient Clip:       {CLIP}")
print(f"Teacher Forcing:     {TEACHER_FORCING_RATIO}")
print(f"Early Stopping:      patience={PATIENCE}")
print(f"Batch Size:          {BATCH_SIZE}")
print("=" * 60)


# In[ ]:


# ==============================================================================
# CELL 4.4: VÒNG LẶP HUẤN LUYỆN ATTENTION MODEL (VỚI EARLY STOPPING)
# ==============================================================================

# Biến theo dõi
best_valid_loss = float('inf')
epochs_without_improvement = 0
attention_history = {
    'train_loss': [],
    'valid_loss': [],
    'train_ppl': [],
    'valid_ppl': []
}

print("=" * 60)
print("🚀 BẮT ĐẦU HUẤN LUYỆN ATTENTION MODEL")
print("=" * 60 + "\n")

for epoch in range(N_EPOCHS):
    
    start_time = time.time()
    
    # ===== TRAIN =====
    attention_model.train()
    train_loss = 0
    
    for batch in tqdm(train_loader, desc=f"Attention Epoch {epoch+1}/{N_EPOCHS}", leave=False):
        src, trg, src_len = batch
        src = src.to(device)
        trg = trg.to(device)
        
        attention_optimizer.zero_grad()
        
        output = attention_model(src, src_len, trg, TEACHER_FORCING_RATIO)
        
        output_dim = output.shape[-1]
        output = output[1:].reshape(-1, output_dim)
        trg = trg[1:].reshape(-1)
        
        loss = attention_criterion(output, trg)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(attention_model.parameters(), CLIP)
        attention_optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    
    # ===== EVALUATE =====
    attention_model.eval()
    valid_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(valid_loader, desc="Validating", leave=False):
            src, trg, src_len = batch
            src = src.to(device)
            trg = trg.to(device)
            
            output = attention_model(src, src_len, trg, 0)
            
            output_dim = output.shape[-1]
            output = output[1:].reshape(-1, output_dim)
            trg = trg[1:].reshape(-1)
            
            loss = attention_criterion(output, trg)
            valid_loss += loss.item()
    
    valid_loss /= len(valid_loader)
    
    end_time = time.time()
    epoch_mins = int((end_time - start_time) / 60)
    epoch_secs = int((end_time - start_time) % 60)
    
    # ===== TÍNH PERPLEXITY =====
    train_ppl = math.exp(train_loss)
    valid_ppl = math.exp(valid_loss)
    
    # Lưu history
    attention_history['train_loss'].append(train_loss)
    attention_history['valid_loss'].append(valid_loss)
    attention_history['train_ppl'].append(train_ppl)
    attention_history['valid_ppl'].append(valid_ppl)
    
    # ===== CHECKPOINTING & EARLY STOPPING =====
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        epochs_without_improvement = 0
        
        # Lưu best model
        torch.save(attention_model.state_dict(), 'attention_best_model.pth')
        save_status = "✅ Model saved!"
    else:
        epochs_without_improvement += 1
        save_status = f"⚠️ No improvement ({epochs_without_improvement}/{PATIENCE})"
    
    # ===== IN KẾT QUẢ =====
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {train_ppl:7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {valid_ppl:7.3f}')
    print(f'\t{save_status}')
    print("-" * 60)
    
    # ===== KIỂM TRA EARLY STOPPING =====
    if epochs_without_improvement >= PATIENCE:
        print("\n" + "=" * 60)
        print(f"⛔ EARLY STOPPING: Val loss không giảm sau {PATIENCE} epochs")
        print("=" * 60)
        break

# =============================================================================
# TỔNG KẾT
# =============================================================================
print("\n" + "=" * 60)
print("🎉 ATTENTION TRAINING HOÀN TẤT!")
print("=" * 60)
print(f"Epochs đã chạy:       {epoch + 1}")
print(f"Best Validation Loss: {best_valid_loss:.3f}")
print(f"Best Validation PPL:  {math.exp(best_valid_loss):.3f}")
print(f"Model đã lưu tại:     'attention_best_model.pth'")
print("=" * 60)


# In[ ]:


# ==============================================================================
# CELL 4.5: SO SÁNH KẾT QUẢ BASELINE VS ATTENTION
# ==============================================================================

print("=" * 60)
print("📊 SO SÁNH KẾT QUẢ TRAINING: BASELINE VS ATTENTION")
print("=" * 60)

# Load best models
baseline_model.load_state_dict(torch.load('baseline_model.pth', map_location=device, weights_only=True))
attention_model.load_state_dict(torch.load('attention_best_model.pth', map_location=device, weights_only=True))

# Evaluate on test set
def evaluate_model(model, iterator, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in iterator:
            src, trg, src_len = batch
            src = src.to(device)
            trg = trg.to(device)
            output = model(src, src_len, trg, 0)
            output_dim = output.shape[-1]
            output = output[1:].reshape(-1, output_dim)
            trg = trg[1:].reshape(-1)
            loss = criterion(output, trg)
            total_loss += loss.item()
    return total_loss / len(iterator)

# Evaluate both models
baseline_test_loss = evaluate_model(baseline_model, test_loader, baseline_criterion, device)
attention_test_loss = evaluate_model(attention_model, test_loader, attention_criterion, device)

print("\n" + "-" * 60)
print("📌 BASELINE SEQ2SEQ (Fixed Context Vector):")
print(f"   Test Loss: {baseline_test_loss:.3f} | Test PPL: {math.exp(baseline_test_loss):7.3f}")
print("\n" + "-" * 60)
print("🎯 ATTENTION SEQ2SEQ (Dynamic Context Vector):")
print(f"   Test Loss: {attention_test_loss:.3f} | Test PPL: {math.exp(attention_test_loss):7.3f}")
print("\n" + "-" * 60)

# Improvement
improvement = baseline_test_loss - attention_test_loss
ppl_improvement = math.exp(baseline_test_loss) - math.exp(attention_test_loss)
print(f"📈 Cải thiện khi dùng Attention:")
print(f"   Loss giảm:       {improvement:.3f}")
print(f"   Perplexity giảm: {ppl_improvement:.3f}")
print("=" * 60)
print("\n✅ PHẦN 4 HOÀN TẤT - Sẵn sàng cho PHẦN 5 (Inference & Evaluation)!")


# # 📊 PHẦN 5: INFERENCE & BLEU EVALUATION
# 
# ---
# 
# ## Quy trình Inference
# 
# ```
# ┌─────────────────────────────────────────────────────────────────────┐
# │                        GREEDY DECODING                              │
# ├─────────────────────────────────────────────────────────────────────┤
# │                                                                     │
# │  Input (EN): "A man walks"                                          │
# │       ↓                                                             │
# │  Tokenize + Numericalize                                            │
# │       ↓                                                             │
# │  Encoder → (encoder_outputs, hidden, cell)                          │
# │       ↓                                                             │
# │  Loop until <eos> or MAX_LEN:                                       │
# │       ├── Decoder(input, hidden, cell, encoder_outputs, mask)       │
# │       ├── argmax(output) → predicted token                          │
# │       └── Append to result                                          │
# │       ↓                                                             │
# │  Output (FR): "Un homme marche"                                     │
# │                                                                     │
# └─────────────────────────────────────────────────────────────────────┘
# ```
# 
# **BLEU Score Evaluation:**
# - Sử dụng `nltk.translate.bleu_score`
# - Smoothing function để xử lý n-gram = 0
# - Đánh giá trên toàn bộ tập Test
# 
# ---

# In[ ]:


# ==============================================================================
# CELL 5.1: HÀM TRANSLATE (CHO CẢ BASELINE VÀ ATTENTION)
# ==============================================================================

def translate_attention(sentence: str) -> str:
    """
    Dịch một câu tiếng Anh sang tiếng Pháp bằng Attention Model.
    Sử dụng Greedy Decoding.
    """
    MAX_LEN = 50
    attention_model.eval()
    
    # Tokenize
    tokens = tokenizer_en(sentence.lower())
    tokens = ['<sos>'] + tokens + ['<eos>']
    
    # Numericalize
    src_indexes = [vocab_en[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)
    src_len = torch.tensor([len(src_indexes)], dtype=torch.long)
    
    # Encoder forward
    with torch.no_grad():
        encoder_outputs, hidden, cell = attention_model.encoder(src_tensor, src_len)
    
    # Mask
    mask = (src_tensor != PAD_IDX).permute(1, 0)
    
    # Greedy decoding
    trg_indexes = [SOS_IDX]
    
    for _ in range(MAX_LEN):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
        
        with torch.no_grad():
            output, hidden, cell, attention = attention_model.decoder(
                trg_tensor, hidden, cell, encoder_outputs, mask
            )
        
        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)
        
        if pred_token == EOS_IDX:
            break
    
    # Convert to words
    trg_tokens = [vocab_fr.get_itos()[i] for i in trg_indexes]
    
    if trg_tokens[0] == '<sos>':
        trg_tokens = trg_tokens[1:]
    if '<eos>' in trg_tokens:
        trg_tokens = trg_tokens[:trg_tokens.index('<eos>')]
    
    return ' '.join(trg_tokens)


def translate_baseline(sentence: str) -> str:
    """
    Dịch một câu tiếng Anh sang tiếng Pháp bằng Baseline Model.
    Sử dụng Greedy Decoding với FIXED context vector.
    """
    MAX_LEN = 50
    baseline_model.eval()
    
    # Tokenize
    tokens = tokenizer_en(sentence.lower())
    tokens = ['<sos>'] + tokens + ['<eos>']
    
    # Numericalize
    src_indexes = [vocab_en[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)
    src_len = torch.tensor([len(src_indexes)], dtype=torch.long)
    
    # Encoder forward (baseline chỉ trả về hidden, cell)
    with torch.no_grad():
        hidden, cell = baseline_model.encoder(src_tensor, src_len)
    
    # Greedy decoding
    trg_indexes = [SOS_IDX]
    
    for _ in range(MAX_LEN):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
        
        with torch.no_grad():
            output, hidden, cell = baseline_model.decoder(
                trg_tensor, hidden, cell
            )
        
        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)
        
        if pred_token == EOS_IDX:
            break
    
    # Convert to words
    trg_tokens = [vocab_fr.get_itos()[i] for i in trg_indexes]
    
    if trg_tokens[0] == '<sos>':
        trg_tokens = trg_tokens[1:]
    if '<eos>' in trg_tokens:
        trg_tokens = trg_tokens[:trg_tokens.index('<eos>')]
    
    return ' '.join(trg_tokens)


# Alias cho backward compatibility
translate = translate_attention

print("✅ Đã định nghĩa hàm translate_attention() và translate_baseline()")


# In[ ]:


# ==============================================================================
# CELL 5.2: TÍNH BLEU SCORE (NLTK)
# ==============================================================================

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

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
    # Smoothing để xử lý n-gram = 0
    smooth = SmoothingFunction().method1
    
    total_bleu = 0
    count = 0
    
    # Giới hạn số mẫu nếu cần
    if num_samples:
        indices = random.sample(range(len(test_src)), min(num_samples, len(test_src)))
    else:
        indices = range(len(test_src))
    
    print("📊 Đang tính BLEU Score...")
    
    for idx in tqdm(indices, desc="Calculating BLEU"):
        src_sentence = test_src[idx]
        trg_sentence = test_trg[idx]
        
        # Dịch câu
        pred_sentence = translate(src_sentence)
        
        # Tokenize
        pred_tokens = pred_sentence.split()
        ref_tokens = tokenizer_fr(trg_sentence.lower())
        
        # NLTK sentence_bleu format
        reference = [ref_tokens]  # List of list
        hypothesis = pred_tokens
        
        try:
            bleu = sentence_bleu(reference, hypothesis, smoothing_function=smooth)
            total_bleu += bleu
            count += 1
        except:
            continue
    
    bleu_avg = total_bleu / count if count > 0 else 0
    return bleu_avg


def demo_translation(test_src, test_trg, num_examples=5):
    """
    Demo dịch một số câu ngẫu nhiên từ tập Test.
    """
    print("\n" + "=" * 70)
    print("🔍 DEMO DỊCH MẪU")
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


print("✅ Đã định nghĩa: calculate_bleu_score(), demo_translation()")


# In[ ]:


# ==============================================================================
# CELL 5.3: DEMO SO SÁNH BASELINE VS ATTENTION
# ==============================================================================

print("=" * 70)
print("📊 DEMO SO SÁNH: BASELINE VS ATTENTION")
print("=" * 70)

# Load best models
baseline_model.load_state_dict(torch.load('baseline_model.pth', map_location=device, weights_only=True))
attention_model.load_state_dict(torch.load('attention_best_model.pth', map_location=device, weights_only=True))
baseline_model.eval()
attention_model.eval()
print("✅ Đã load cả hai models\n")

# Demo 5 câu ngẫu nhiên
indices = random.sample(range(len(test_en)), 5)

for i, idx in enumerate(indices, 1):
    src = test_en[idx]
    trg = test_fr[idx]
    pred_baseline = translate_baseline(src)
    pred_attention = translate_attention(src)
    
    print(f"\n--- Ví dụ {i} ---")
    print(f"📥 Source (EN):     {src}")
    print(f"📌 Reference (FR):  {trg}")
    print(f"🏗️ Baseline (FR):   {pred_baseline}")
    print(f"🎯 Attention (FR):  {pred_attention}")

print("\n" + "=" * 70)


# In[ ]:


# ==============================================================================
# CELL 5.4: ĐÁNH GIÁ BLEU SCORE - BASELINE VS ATTENTION
# ==============================================================================

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def calculate_bleu(model_translate_fn, test_src, test_trg, num_samples=None):
    """
    Tính điểm BLEU trung bình cho một hàm translate.
    """
    smooth = SmoothingFunction().method1
    total_bleu = 0
    count = 0
    
    if num_samples:
        indices = random.sample(range(len(test_src)), min(num_samples, len(test_src)))
    else:
        indices = range(len(test_src))
    
    for idx in tqdm(indices, desc="Calculating BLEU"):
        src_sentence = test_src[idx]
        trg_sentence = test_trg[idx]
        
        pred_sentence = model_translate_fn(src_sentence)
        
        pred_tokens = pred_sentence.split()
        ref_tokens = tokenizer_fr(trg_sentence.lower())
        
        reference = [ref_tokens]
        hypothesis = pred_tokens
        
        try:
            bleu = sentence_bleu(reference, hypothesis, smoothing_function=smooth)
            total_bleu += bleu
            count += 1
        except:
            continue
    
    return total_bleu / count if count > 0 else 0


print("=" * 70)
print("📊 ĐÁNH GIÁ BLEU SCORE: BASELINE VS ATTENTION")
print("=" * 70)

# Tính BLEU cho Baseline
print("\n🏗️ Tính BLEU cho BASELINE...")
baseline_bleu = calculate_bleu(translate_baseline, test_en, test_fr, num_samples=None)

# Tính BLEU cho Attention
print("\n🎯 Tính BLEU cho ATTENTION...")
attention_bleu = calculate_bleu(translate_attention, test_en, test_fr, num_samples=None)

print("\n" + "=" * 70)
print("📊 KẾT QUẢ BLEU SCORE")
print("=" * 70)
print(f"🏗️ BASELINE (Fixed Context):   {baseline_bleu * 100:.2f}%")
print(f"🎯 ATTENTION (Dynamic Context): {attention_bleu * 100:.2f}%")
print("-" * 70)
print(f"📈 Cải thiện với Attention:     {(attention_bleu - baseline_bleu) * 100:.2f}%")
print("=" * 70)

# Bảng đánh giá
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

📌 Kết luận: 
- Baseline Seq2Seq với Fixed Context Vector hoạt động, nhưng hạn chế.
- Attention Seq2Seq cải thiện đáng kể nhờ Dynamic Context Vector.
""")


# In[ ]:


# ==============================================================================
# CELL 5.5: DỊCH CÂU TÙY Ý
# ==============================================================================

print("=" * 70)
print("🖊️ DỊCH CÂU TÙY Ý")
print("=" * 70)

test_sentences = [
    "I love machine learning.",
    "The weather is beautiful today.",
    "A man is walking with his dog.",
    "Two children are playing in the park.",
    "A woman is reading a book.",
]

for sentence in test_sentences:
    result = translate(sentence)
    print(f"\n📥 EN: {sentence}")
    print(f"🇫🇷 FR: {result}")

print("\n" + "=" * 70)
print("\n✅ PHẦN 5 HOÀN TẤT - Sẵn sàng cho PHẦN 6 (Analysis)!")


# In[ ]:


# ==============================================================================
# CELL 4.6: VẼ BIỂU ĐỒ TRAINING HISTORY (Optional)
# ==============================================================================

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot Loss
axes[0].plot(training_history['train_loss'], label='Train Loss', marker='o')
axes[0].plot(training_history['valid_loss'], label='Valid Loss', marker='s')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training & Validation Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot Perplexity
axes[1].plot(training_history['train_ppl'], label='Train PPL', marker='o')
axes[1].plot(training_history['valid_ppl'], label='Valid PPL', marker='s')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Perplexity')
axes[1].set_title('Training & Validation Perplexity')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150)
plt.show()

print("✅ Đã lưu biểu đồ tại 'training_history.png'")


# # 📊 PHẦN 6: CÁC HÀM BỔ SUNG CHO BÁO CÁO
# 
# ---
# 
# ## Các công cụ phân tích nâng cao
# 
# **Bao gồm:**
# 1. `plot_history()` - Vẽ biểu đồ Loss với highlight epoch tốt nhất
# 2. `analyze_model_performance()` - Phân tích lỗi với BLEU score
# 3. `translate_beam_search()` - Beam Search decoding (điểm cộng)
# 
# ---

# In[ ]:


# ==============================================================================
# CELL 6.1: HÀM VẼ BIỂU ĐỒ LOSS
# ==============================================================================

import matplotlib.pyplot as plt

def plot_history(history, title="Training History", filename="training_history.png"):
    """
    Vẽ biểu đồ Train Loss và Valid Loss từ history.
    
    Args:
        history: Dict chứa 'train_loss' và 'valid_loss' (list)
        title: Tiêu đề biểu đồ
        filename: Tên file PNG để lưu
    
    Returns:
        None (lưu file PNG)
    """
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.plot(epochs, history['train_loss'], 'b-o', label='Train Loss', linewidth=2, markersize=6)
    plt.plot(epochs, history['valid_loss'], 'r-s', label='Valid Loss', linewidth=2, markersize=6)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(epochs)
    
    # Highlight best epoch
    best_epoch = history['valid_loss'].index(min(history['valid_loss'])) + 1
    best_loss = min(history['valid_loss'])
    plt.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, label=f'Best: Epoch {best_epoch}')
    plt.scatter([best_epoch], [best_loss], color='green', s=100, zorder=5, marker='*')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Đã lưu biểu đồ: {filename}")
    print(f"   Best Epoch: {best_epoch} | Best Valid Loss: {best_loss:.4f}")

print("✅ Đã định nghĩa hàm plot_history()")


# In[ ]:


# ==============================================================================
# CELL 6.2: HÀM PHÂN TÍCH HIỆU SUẤT MÔ HÌNH
# ==============================================================================

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def analyze_model_performance(translate_fn, src_sentences, trg_sentences, num_examples=5):
    """
    Phân tích hiệu suất mô hình: tìm câu BLEU cao nhất và thấp nhất.
    
    Args:
        translate_fn: Hàm dịch (translate_attention hoặc translate_baseline)
        src_sentences: List câu nguồn (tiếng Anh)
        trg_sentences: List câu đích ground truth (tiếng Pháp)
        num_examples: Số câu hiển thị cho mỗi nhóm (cao/thấp)
    
    Returns:
        Dict chứa 'best' và 'worst' examples
    """
    smooth = SmoothingFunction().method1
    results = []
    
    print("=" * 80)
    print("📊 PHÂN TÍCH HIỆU SUẤT MÔ HÌNH")
    print("=" * 80)
    print(f"Đang đánh giá {len(src_sentences)} câu...")
    
    for idx in tqdm(range(len(src_sentences)), desc="Analyzing"):
        src = src_sentences[idx]
        trg = trg_sentences[idx]
        pred = translate_fn(src)
        
        # Tính BLEU
        pred_tokens = pred.split()
        ref_tokens = tokenizer_fr(trg.lower())
        
        try:
            bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth)
        except:
            bleu = 0.0
        
        results.append({
            'idx': idx,
            'src': src,
            'trg': trg,
            'pred': pred,
            'bleu': bleu
        })
    
    # Sắp xếp theo BLEU
    results_sorted = sorted(results, key=lambda x: x['bleu'], reverse=True)
    
    best_examples = results_sorted[:num_examples]
    worst_examples = results_sorted[-num_examples:]
    
    # Tính thống kê
    bleu_scores = [r['bleu'] for r in results]
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    
    print("\n" + "-" * 80)
    print(f"📈 THỐNG KÊ TỔNG QUAN")
    print("-" * 80)
    print(f"   Tổng số câu:     {len(results)}")
    print(f"   BLEU trung bình: {avg_bleu * 100:.2f}%")
    print(f"   BLEU cao nhất:   {max(bleu_scores) * 100:.2f}%")
    print(f"   BLEU thấp nhất:  {min(bleu_scores) * 100:.2f}%")
    
    # In câu BLEU cao nhất
    print("\n" + "=" * 80)
    print(f"🏆 TOP {num_examples} CÂU BLEU CAO NHẤT")
    print("=" * 80)
    
    for i, ex in enumerate(best_examples, 1):
        print(f"\n--- Rank {i} | BLEU: {ex['bleu'] * 100:.2f}% ---")
        print(f"   EN (Source):    {ex['src']}")
        print(f"   FR (Reference): {ex['trg']}")
        print(f"   FR (Predicted): {ex['pred']}")
    
    # In câu BLEU thấp nhất
    print("\n" + "=" * 80)
    print(f"⚠️ TOP {num_examples} CÂU BLEU THẤP NHẤT")
    print("=" * 80)
    
    for i, ex in enumerate(worst_examples, 1):
        print(f"\n--- Rank {len(results) - num_examples + i} | BLEU: {ex['bleu'] * 100:.2f}% ---")
        print(f"   EN (Source):    {ex['src']}")
        print(f"   FR (Reference): {ex['trg']}")
        print(f"   FR (Predicted): {ex['pred']}")
    
    print("\n" + "=" * 80)
    
    return {
        'avg_bleu': avg_bleu,
        'best': best_examples,
        'worst': worst_examples,
        'all_results': results
    }

print("✅ Đã định nghĩa hàm analyze_model_performance()")


# In[ ]:


# ==============================================================================
# CELL 6.3: HÀM BEAM SEARCH DECODING (ĐIỂM CỘNG)
# ==============================================================================

def translate_beam_search(sentence, model, beam_size=3, max_len=50):
    """
    Dịch câu sử dụng Beam Search Decoding (cho Attention Model).
    
    Args:
        sentence: Câu tiếng Anh cần dịch
        model: Attention model (Seq2SeqAttention)
        beam_size: Số beam giữ lại mỗi bước
        max_len: Độ dài tối đa của câu dịch
    
    Returns:
        str: Câu tiếng Pháp được dịch
    """
    model.eval()
    
    # Tokenize và numericalize
    tokens = tokenizer_en(sentence.lower())
    tokens = ['<sos>'] + tokens + ['<eos>']
    src_indexes = [vocab_en[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)
    src_len = torch.tensor([len(src_indexes)], dtype=torch.long)
    
    # Encode
    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src_tensor, src_len)
    
    # Mask
    mask = (src_tensor != PAD_IDX).permute(1, 0)
    
    # Beam Search initialization
    beams = [(0.0, [SOS_IDX], hidden, cell)]
    completed = []
    
    for _ in range(max_len):
        new_beams = []
        
        for score, tokens, h, c in beams:
            if tokens[-1] == EOS_IDX:
                completed.append((score, tokens))
                continue
            
            trg_tensor = torch.LongTensor([tokens[-1]]).to(device)
            
            with torch.no_grad():
                output, new_h, new_c, _ = model.decoder(trg_tensor, h, c, encoder_outputs, mask)
            
            # Log probabilities
            log_probs = F.log_softmax(output, dim=1)
            
            # Top-k candidates
            topk_log_probs, topk_indices = log_probs.topk(beam_size)
            
            for i in range(beam_size):
                new_score = score + topk_log_probs[0, i].item()
                new_token = topk_indices[0, i].item()
                new_tokens = tokens + [new_token]
                new_beams.append((new_score, new_tokens, new_h, new_c))
        
        # Keep top beams
        new_beams.sort(key=lambda x: x[0], reverse=True)
        beams = new_beams[:beam_size]
        
        if len(beams) == 0:
            break
    
    # Add remaining beams to completed
    for score, tokens, h, c in beams:
        completed.append((score, tokens))
    
    # Chọn beam có score cao nhất (normalize by length)
    if completed:
        completed.sort(key=lambda x: x[0] / len(x[1]), reverse=True)
        best_tokens = completed[0][1]
    else:
        best_tokens = [SOS_IDX]
    
    # Convert to words
    trg_tokens = [vocab_fr.get_itos()[i] for i in best_tokens]
    
    if trg_tokens[0] == '<sos>':
        trg_tokens = trg_tokens[1:]
    if '<eos>' in trg_tokens:
        trg_tokens = trg_tokens[:trg_tokens.index('<eos>')]
    
    return ' '.join(trg_tokens)

print("✅ Đã định nghĩa hàm translate_beam_search()")


# In[ ]:


# ==============================================================================
# CELL 6.4: DEMO SỬ DỤNG CÁC HÀM MỚI
# ==============================================================================

print("=" * 80)
print("🚀 DEMO CÁC HÀM BỔ SUNG")
print("=" * 80)

# 1. Vẽ biểu đồ Loss
print("\n📊 1. VẼ BIỂU ĐỒ LOSS")
print("-" * 40)

if baseline_history['train_loss']:
    plot_history(baseline_history, "Baseline Seq2Seq", "baseline_loss.png")

if attention_history['train_loss']:
    plot_history(attention_history, "Attention Seq2Seq", "attention_loss.png")

# 2. Phân tích lỗi (trên subset nhỏ để demo)
print("\n📊 2. PHÂN TÍCH LỖI (DEMO - 100 CÂU)")
print("-" * 40)

# Load best models
baseline_model.load_state_dict(torch.load('baseline_model.pth', map_location=device, weights_only=True))
attention_model.load_state_dict(torch.load('attention_best_model.pth', map_location=device, weights_only=True))

print("\n🎯 Phân tích ATTENTION MODEL:")
attention_analysis = analyze_model_performance(
    translate_attention, 
    test_en[:100],  # Chỉ 100 câu để demo
    test_fr[:100], 
    num_examples=3
)

# 3. Demo Beam Search
print("\n📊 3. DEMO BEAM SEARCH")
print("-" * 40)

demo_sentences = [
    "A man is walking with his dog.",
    "Two children are playing in the park."
]

print("\n🔍 So sánh Greedy vs Beam Search:")
for sent in demo_sentences:
    greedy = translate_attention(sent)
    beam = translate_beam_search(sent, attention_model, beam_size=3)
    
    print(f"\n📥 EN: {sent}")
    print(f"   Greedy:      {greedy}")
    print(f"   Beam (k=3):  {beam}")

print("\n" + "=" * 80)
print("✅ HOÀN TẤT TẤT CẢ DEMO!")
print("=" * 80)


# # 📊 PHẦN 6: CÁC HÀM BỔ SUNG CHO BÁO CÁO
# 
# ---
# 
# ## Các công cụ phân tích nâng cao
# 
# **Bao gồm:**
# 1. `plot_history()` - Vẽ biểu đồ Loss với highlight epoch tốt nhất
# 2. `analyze_model_performance()` - Phân tích lỗi với BLEU score
# 3. `translate_beam_search()` - Beam Search decoding (điểm cộng)
# 
# ---

# In[ ]:


# ==============================================================================
# CELL 6.1: HÀM VẼ BIỂU ĐỒ LOSS
# ==============================================================================

import matplotlib.pyplot as plt

def plot_history(history, title="Training History", filename="training_history.png"):
    """
    Vẽ biểu đồ Train Loss và Valid Loss từ history.
    
    Args:
        history: Dict chứa 'train_loss' và 'valid_loss' (list)
        title: Tiêu đề biểu đồ
        filename: Tên file PNG để lưu
    
    Returns:
        None (lưu file PNG)
    """
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.plot(epochs, history['train_loss'], 'b-o', label='Train Loss', linewidth=2, markersize=6)
    plt.plot(epochs, history['valid_loss'], 'r-s', label='Valid Loss', linewidth=2, markersize=6)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(epochs)
    
    # Highlight best epoch
    best_epoch = history['valid_loss'].index(min(history['valid_loss'])) + 1
    best_loss = min(history['valid_loss'])
    plt.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, label=f'Best: Epoch {best_epoch}')
    plt.scatter([best_epoch], [best_loss], color='green', s=100, zorder=5, marker='*')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Đã lưu biểu đồ: {filename}")
    print(f"   Best Epoch: {best_epoch} | Best Valid Loss: {best_loss:.4f}")

print("✅ Đã định nghĩa hàm plot_history()")


# In[ ]:


# ==============================================================================
# CELL 6.2: HÀM PHÂN TÍCH HIỆU SUẤT MÔ HÌNH
# ==============================================================================

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def analyze_model_performance(translate_fn, src_sentences, trg_sentences, num_examples=5):
    """
    Phân tích hiệu suất mô hình: tìm câu BLEU cao nhất và thấp nhất.
    
    Args:
        translate_fn: Hàm dịch (translate_attention hoặc translate_baseline)
        src_sentences: List câu nguồn (tiếng Anh)
        trg_sentences: List câu đích ground truth (tiếng Pháp)
        num_examples: Số câu hiển thị cho mỗi nhóm (cao/thấp)
    
    Returns:
        Dict chứa 'best' và 'worst' examples
    """
    smooth = SmoothingFunction().method1
    results = []
    
    print("=" * 80)
    print("📊 PHÂN TÍCH HIỆU SUẤT MÔ HÌNH")
    print("=" * 80)
    print(f"Đang đánh giá {len(src_sentences)} câu...")
    
    for idx in tqdm(range(len(src_sentences)), desc="Analyzing"):
        src = src_sentences[idx]
        trg = trg_sentences[idx]
        pred = translate_fn(src)
        
        # Tính BLEU
        pred_tokens = pred.split()
        ref_tokens = tokenizer_fr(trg.lower())
        
        try:
            bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth)
        except:
            bleu = 0.0
        
        results.append({
            'idx': idx,
            'src': src,
            'trg': trg,
            'pred': pred,
            'bleu': bleu
        })
    
    # Sắp xếp theo BLEU
    results_sorted = sorted(results, key=lambda x: x['bleu'], reverse=True)
    
    best_examples = results_sorted[:num_examples]
    worst_examples = results_sorted[-num_examples:]
    
    # Tính thống kê
    bleu_scores = [r['bleu'] for r in results]
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    
    print("\n" + "-" * 80)
    print(f"📈 THỐNG KÊ TỔNG QUAN")
    print("-" * 80)
    print(f"   Tổng số câu:     {len(results)}")
    print(f"   BLEU trung bình: {avg_bleu * 100:.2f}%")
    print(f"   BLEU cao nhất:   {max(bleu_scores) * 100:.2f}%")
    print(f"   BLEU thấp nhất:  {min(bleu_scores) * 100:.2f}%")
    
    # In câu BLEU cao nhất
    print("\n" + "=" * 80)
    print(f"🏆 TOP {num_examples} CÂU BLEU CAO NHẤT")
    print("=" * 80)
    
    for i, ex in enumerate(best_examples, 1):
        print(f"\n--- Rank {i} | BLEU: {ex['bleu'] * 100:.2f}% ---")
        print(f"   EN (Source):    {ex['src']}")
        print(f"   FR (Reference): {ex['trg']}")
        print(f"   FR (Predicted): {ex['pred']}")
    
    # In câu BLEU thấp nhất
    print("\n" + "=" * 80)
    print(f"⚠️ TOP {num_examples} CÂU BLEU THẤP NHẤT")
    print("=" * 80)
    
    for i, ex in enumerate(worst_examples, 1):
        print(f"\n--- Rank {len(results) - num_examples + i} | BLEU: {ex['bleu'] * 100:.2f}% ---")
        print(f"   EN (Source):    {ex['src']}")
        print(f"   FR (Reference): {ex['trg']}")
        print(f"   FR (Predicted): {ex['pred']}")
    
    print("\n" + "=" * 80)
    
    return {
        'avg_bleu': avg_bleu,
        'best': best_examples,
        'worst': worst_examples,
        'all_results': results
    }

print("✅ Đã định nghĩa hàm analyze_model_performance()")


# In[ ]:


# ==============================================================================
# CELL 6.3: HÀM BEAM SEARCH DECODING (ĐIỂM CỘNG)
# ==============================================================================

def translate_beam_search(sentence, model, beam_size=3, max_len=50):
    """
    Dịch câu sử dụng Beam Search Decoding (cho Attention Model).
    
    Args:
        sentence: Câu tiếng Anh cần dịch
        model: Attention model (Seq2SeqAttention)
        beam_size: Số beam giữ lại mỗi bước
        max_len: Độ dài tối đa của câu dịch
    
    Returns:
        str: Câu tiếng Pháp được dịch
    """
    model.eval()
    
    # Tokenize và numericalize
    tokens = tokenizer_en(sentence.lower())
    tokens = ['<sos>'] + tokens + ['<eos>']
    src_indexes = [vocab_en[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)
    src_len = torch.tensor([len(src_indexes)], dtype=torch.long)
    
    # Encode
    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src_tensor, src_len)
    
    # Mask
    mask = (src_tensor != PAD_IDX).permute(1, 0)
    
    # Beam Search initialization
    beams = [(0.0, [SOS_IDX], hidden, cell)]
    completed = []
    
    for _ in range(max_len):
        new_beams = []
        
        for score, tokens, h, c in beams:
            if tokens[-1] == EOS_IDX:
                completed.append((score, tokens))
                continue
            
            trg_tensor = torch.LongTensor([tokens[-1]]).to(device)
            
            with torch.no_grad():
                output, new_h, new_c, _ = model.decoder(trg_tensor, h, c, encoder_outputs, mask)
            
            # Log probabilities
            log_probs = F.log_softmax(output, dim=1)
            
            # Top-k candidates
            topk_log_probs, topk_indices = log_probs.topk(beam_size)
            
            for i in range(beam_size):
                new_score = score + topk_log_probs[0, i].item()
                new_token = topk_indices[0, i].item()
                new_tokens = tokens + [new_token]
                new_beams.append((new_score, new_tokens, new_h, new_c))
        
        # Keep top beams
        new_beams.sort(key=lambda x: x[0], reverse=True)
        beams = new_beams[:beam_size]
        
        if len(beams) == 0:
            break
    
    # Add remaining beams to completed
    for score, tokens, h, c in beams:
        completed.append((score, tokens))
    
    # Chọn beam có score cao nhất (normalize by length)
    if completed:
        completed.sort(key=lambda x: x[0] / len(x[1]), reverse=True)
        best_tokens = completed[0][1]
    else:
        best_tokens = [SOS_IDX]
    
    # Convert to words
    trg_tokens = [vocab_fr.get_itos()[i] for i in best_tokens]
    
    if trg_tokens[0] == '<sos>':
        trg_tokens = trg_tokens[1:]
    if '<eos>' in trg_tokens:
        trg_tokens = trg_tokens[:trg_tokens.index('<eos>')]
    
    return ' '.join(trg_tokens)

print("✅ Đã định nghĩa hàm translate_beam_search()")
# In[ ]:


# ==============================================================================
# CELL 6.4: DEMO SỬ DỤNG CÁC HÀM MỚI
# ==============================================================================

print("=" * 80)
print("🚀 DEMO CÁC HÀM BỔ SUNG")
print("=" * 80)

# 1. Vẽ biểu đồ Loss
print("\n📊 1. VẼ BIỂU ĐỒ LOSS")
print("-" * 40)

if baseline_history['train_loss']:
    plot_history(baseline_history, "Baseline Seq2Seq", "baseline_loss.png")

if attention_history['train_loss']:
    plot_history(attention_history, "Attention Seq2Seq", "attention_loss.png")

# 2. Phân tích lỗi (trên subset nhỏ để demo)
print("\n📊 2. PHÂN TÍCH LỖI (DEMO - 100 CÂU)")
print("-" * 40)

# Load best models
baseline_model.load_state_dict(torch.load('baseline_model.pth', map_location=device, weights_only=True))
attention_model.load_state_dict(torch.load('attention_best_model.pth', map_location=device, weights_only=True))

print("\n🎯 Phân tích ATTENTION MODEL:")
attention_analysis = analyze_model_performance(
    translate_attention, 
    test_en[:100],  # Chỉ 100 câu để demo
    test_fr[:100], 
    num_examples=3
)

# 3. Demo Beam Search
print("\n📊 3. DEMO BEAM SEARCH")
print("-" * 40)

demo_sentences = [
    "A man is walking with his dog.",
    "Two children are playing in the park."
]

print("\n🔍 So sánh Greedy vs Beam Search:")
for sent in demo_sentences:
    greedy = translate_attention(sent)
    beam = translate_beam_search(sent, attention_model, beam_size=3)
    
    print(f"\n📥 EN: {sent}")
    print(f"   Greedy:      {greedy}")
    print(f"   Beam (k=3):  {beam}")

print("\n" + "=" * 80)
print("✅ HOÀN TẤT TẤT CẢ DEMO!")
print("=" * 80)

