"""
==============================================================================
CODE ĐÃ REFACTOR - COPY VÀO NOTEBOOK
==============================================================================
Thay thế các cell tương ứng trong nlp (1).ipynb

📌 HƯỚNG DẪN:
1. Copy code từ "CELL 1" vào cell chứa collate_fn
2. Copy code từ "CELL 2" vào cell chứa Encoder, Decoder, Seq2Seq
==============================================================================
"""

# ==============================================================================
# CELL 1: COLLATE FUNCTION (Thay thế cell ~line 395-493)
# ==============================================================================

# CLASS DATASET (Giữ nguyên)
class TranslationDataset(Dataset):
    def __init__(self, src_list, trg_list):
        self.src_list = src_list
        self.trg_list = trg_list

    def __len__(self):
        return len(self.src_list)

    def __getitem__(self, idx):
        return self.src_list[idx], self.trg_list[idx]


# HÀM COLLATE_FN (ĐÃ SỬA)
def collate_fn(batch):
    src_batch, trg_batch = [], []

    # Chuyển đổi Text -> Tensor
    for src_sample, trg_sample in batch:
        src_batch.append(text_transform(src_sample, tokenizer_en, vocab_en))
        trg_batch.append(text_transform(trg_sample, tokenizer_fr, vocab_fr))

    # Padding (Đồng bộ độ dài)
    src_padded = pad_sequence(src_batch, padding_value=PAD_IDX)
    trg_padded = pad_sequence(trg_batch, padding_value=PAD_IDX)

    # ===== FIX 1: Explicit dtype=torch.long =====
    # Đảm bảo kiểu dữ liệu rõ ràng, tránh lỗi tiềm ẩn
    src_lens = torch.tensor([len(x) for x in src_batch], dtype=torch.long)

    # Sort giảm dần
    sorted_lens, sorted_indices = torch.sort(src_lens, descending=True)

    # Sắp xếp lại Tensor theo thứ tự đã sort
    src_padded = src_padded[:, sorted_indices]
    trg_padded = trg_padded[:, sorted_indices]

    # sorted_lens nằm trên CPU (cho pack_padded_sequence)
    return src_padded, trg_padded, sorted_lens


# HÀM GET_DATA_LOADER (Giữ nguyên)
def get_data_loader(dataset, batch_size, collate_fn, shuffle=False):
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      collate_fn=collate_fn)


# ==============================================================================
# CELL 2: MODEL CLASSES (Thay thế cell ~line 547-647)
# ==============================================================================

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len):
        embedded = self.dropout(self.embedding(src))
        # FIX: Luôn gọi .cpu() để đảm bảo an toàn
        packed_embedded = pack_padded_sequence(embedded, src_len.cpu(), enforce_sorted=True)
        packed_outputs, (hidden, cell) = self.lstm(packed_embedded)
        return hidden, cell


class Decoder(nn.Module):
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
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must match!"
        assert encoder.n_layers == decoder.n_layers, \
            "Number of layers of encoder and decoder must match!"

    def forward(self, src, src_len, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        hidden, cell = self.encoder(src, src_len)
        input = trg[0,:]

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            # ===== FIX 2: Dùng torch.rand() thay vì random.random() =====
            # Điều này giúp torch.manual_seed() có tác dụng toàn diện
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1

        return outputs


def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.uniform_(param.data, -0.08, 0.08)
        else:
            nn.init.constant_(param.data, 0)


print("✅ Đã xây dựng xong kiến trúc Model - Phiên bản Optimized!")


# ==============================================================================
# CHANGE LOG (NHẬT KÝ THAY ĐỔI)
# ==============================================================================
"""
┌───┬────────────────────────┬───────────────────────────────────────────────────────┬───────────────────────────────────────────────────────────────┬────────────────────────────────────────────────────────────┐
│ # │ Vị trí                 │ Code Cũ                                               │ Code Mới                                                      │ Lý do                                                      │
├───┼────────────────────────┼───────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────┤
│ 1 │ collate_fn (src_lens)  │ torch.tensor([len(x) for x in src_batch])             │ torch.tensor([len(x) for x in src_batch], dtype=torch.long)   │ Explicit Typing: tránh lỗi tiềm ẩn trên OS/PyTorch khác   │
│ 2 │ Seq2Seq.forward()      │ teacher_force = random.random() < teacher_forcing_ratio│ teacher_force = torch.rand(1).item() < teacher_forcing_ratio │ Reproducibility: torch.manual_seed() có tác dụng toàn diện│
└───┴────────────────────────┴───────────────────────────────────────────────────────┴───────────────────────────────────────────────────────────────┴────────────────────────────────────────────────────────────┘
"""
