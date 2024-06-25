import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torchsummary import summary

# helpers

def pair(t):
    # 입력이 튜플인지 확인하고, 아니면 동일한 값의 튜플로 반환
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        # 피드포워드 네트워크 정의
        self.net = nn.Sequential(
            nn.LayerNorm(dim),  # 입력 정규화, shape: (batch_size, seq_len, dim)
            nn.Linear(dim, hidden_dim),  # 선형 변환, shape: (batch_size, seq_len, hidden_dim)
            nn.GELU(),  # 활성화 함수
            nn.Dropout(dropout),  # 드롭아웃
            nn.Linear(hidden_dim, dim),  # 선형 변환, shape: (batch_size, seq_len, dim)
            nn.Dropout(dropout)  # 드롭아웃
        )

    def forward(self, x):
        # 네트워크를 통해 입력을 처리
        return self.net(x)  # shape: (batch_size, seq_len, dim)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads  # inner_dim = 64 * 8 = 512
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads  # heads = 8
        self.scale = dim_head ** -0.5  # scale = 64 ** -0.5

        self.norm = nn.LayerNorm(dim)  # 입력 정규화, shape: (batch_size, seq_len, dim)

        self.attend = nn.Softmax(dim=-1)  # 소프트맥스 함수
        self.dropout = nn.Dropout(dropout)  # 드롭아웃

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)  # QKV로 변환하는 선형 변환, shape: (batch_size, seq_len, inner_dim * 3)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),  # 출력 선형 변환, shape: (batch_size, seq_len, dim)
            nn.Dropout(dropout)  # 드롭아웃
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)  # 입력 정규화, shape: (batch_size, seq_len, dim)

        # Q, K, V로 변환하고 헤드로 분할
        print(x.shape)
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # shape: (batch_size, seq_len, inner_dim) 각 Q, K, V
        
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)  # shape: (batch_size, heads, seq_len, dim_head) 각 Q, K, V
        print('*' * 50)
        # 어텐션 점수 계산
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # shape: (batch_size, heads, seq_len, seq_len)

        # 소프트맥스를 통해 어텐션 확률 계산
        attn = self.attend(dots)  # shape: (batch_size, heads, seq_len, seq_len)
        attn = self.dropout(attn)  # 드롭아웃 적용

        # 어텐션을 적용하여 출력 계산
        out = torch.matmul(attn, v)  # shape: (batch_size, heads, seq_len, dim_head)
        out = rearrange(out, 'b h n d -> b n (h d)')  # shape: (batch_size, seq_len, inner_dim)
        return self.to_out(out)  # shape: (batch_size, seq_len, dim)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)  # 입력 정규화, shape: (batch_size, seq_len, dim)
        self.layers = nn.ModuleList([])  # 트랜스포머 레이어 리스트
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),  # 어텐션 레이어
                FeedForward(dim, mlp_dim, dropout=dropout)  # 피드포워드 레이어
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x  # 어텐션 레이어 적용 및 잔차 연결, shape: (batch_size, seq_len, dim)
            x = ff(x) + x  # 피드포워드 레이어 적용 및 잔차 연결, shape: (batch_size, seq_len, dim)

        return self.norm(x)  # 출력 정규화, shape: (batch_size, seq_len, dim)

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)  # 이미지 크기, 예: (224, 224)
        patch_height, patch_width = pair(patch_size)  # 패치 크기, 예: (16, 16)

        # 이미지 크기가 패치 크기로 나누어 떨어지는지 확인
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)  # 패치 개수, 예: (14 * 14 = 196)
        patch_dim = channels * patch_height * patch_width  # 패치 차원, 예: (3 * 16 * 16 = 768)
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'  # 풀링 방식 확인

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),  # 이미지를 패치로 나누기, shape: (batch_size, num_patches, patch_dim)
            nn.LayerNorm(patch_dim),  # 패치 정규화, shape: (batch_size, num_patches, patch_dim)
            nn.Linear(patch_dim, dim),  # 패치를 임베딩 차원으로 변환, shape: (batch_size, num_patches, dim)
            nn.LayerNorm(dim)  # 임베딩 정규화, shape: (batch_size, num_patches, dim)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))  # 위치 임베딩, shape: (1, num_patches + 1, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))  # 클래스 토큰, shape: (1, 1, dim)
        self.dropout = nn.Dropout(emb_dropout)  # 드롭아웃

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)  # 트랜스포머 인코더

        self.pool = pool  # 풀링 방식
        self.to_latent = nn.Identity()  # 잠재 공간 변환 (Identity는 그대로 반환)

        self.mlp_head = nn.Linear(dim, num_classes)  # 최종 MLP 헤드, shape: (batch_size, dim) -> (batch_size, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)  # 이미지를 패치 임베딩으로 변환, shape: (batch_size, num_patches, dim)
        b, n, _ = x.shape

        # 각 배치에 동일한 클래스 토큰을 포함
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)  # shape: (batch_size, 1, dim)
        print(x.shape)
        x = torch.cat((cls_tokens, x), dim=1)  # 클래스 토큰을 패치 임베딩 앞에 추가, shape: (batch_size, num_patches + 1, dim)
        print(x.shape)
        x += self.pos_embedding[:, :(n + 1)]  # 위치 임베딩 추가, shape: (batch_size, num_patches + 1, dim)
        x = self.dropout(x)  # 드롭아웃 적용, shape: (batch_size, num_patches + 1, dim)

        x = self.transformer(x)  # 트랜스포머 인코더를 통해 입력 처리, shape: (batch_size, num_patches + 1, dim)
        print(x.shape)
        # 풀링 방식에 따라 평균값 또는 클래스 토큰 선택
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]  # shape: (batch_size, dim)
        print(x.shape)
        x = self.to_latent(x)  # 잠재 공간 변환, shape: (batch_size, dim)
        return self.mlp_head(x)  # MLP 헤드를 통해 최종 출력 반환, shape: (batch_size, num_classes)

    def load_weights(self, path):
        # 사전 학습된 가중치를 로드
        self.load_state_dict(torch.load(path))

# Example usage
if __name__=='__main__':
    model = ViT(
        image_size=224, 
        patch_size=16, 
        num_classes=1000, 
        dim=768, 
        depth=12, 
        heads=12, 
        mlp_dim=3072, 
        pool='cls', 
        channels=3, 
        dim_head=64, 
        dropout=0.1, 
        emb_dropout=0.1
    )

    summary(model, (3, 224, 224))
