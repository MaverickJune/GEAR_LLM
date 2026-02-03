"""
Hugging Face 데이터셋을 로드하고 DataLoader를 생성하는 유틸리티
"""
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import torch
import itertools

'''
import itertools

# 무한 반복
for batch in itertools.cycle(dataloader):
    process(batch)  # 끝나면 자동으로 처음부터 다시 시작
'''

class HuggingFaceDataset(Dataset):
    """Hugging Face 데이터셋을 PyTorch Dataset으로 변환"""
    
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]


def get_hf_dataloader(
    dataset_name: str,
    num_samples: int = 100,
    split: str = "train",
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    **kwargs
):
    """
    Hugging Face에서 데이터셋을 가져와서 PyTorch DataLoader를 반환
    
    Args:
        dataset_name (str): Hugging Face 데이터셋 이름 (예: "squad", "wikitext", "glue")
        num_samples (int): 가져올 데이터 샘플 개수 (기본값: 100)
        split (str): 데이터셋 split ("train", "test", "validation" 등, 기본값: "train")
        batch_size (int): 배치 크기 (기본값: 32)
        shuffle (bool): 데이터 셔플 여부 (기본값: True)
        num_workers (int): DataLoader worker 수 (기본값: 0)
        **kwargs: load_dataset에 전달할 추가 인자 (예: subset name)
    
    Returns:
        DataLoader: PyTorch DataLoader 객체
        
    Examples:
        >>> # SQuAD 데이터셋 100개 샘플 로드
        >>> dataloader = get_hf_dataloader("squad", num_samples=100, split="train")
        
        >>> # WikiText-2 테스트 데이터 50개 샘플 로드
        >>> dataloader = get_hf_dataloader("wikitext", "wikitext-2-raw-v1", 
        ...                                num_samples=50, split="test")
        
        >>> # GLUE의 SST-2 데이터셋 로드
        >>> dataloader = get_hf_dataloader("glue", "sst2", num_samples=200, split="validation")
    """
    try:
        # Hugging Face 데이터셋 로드
        print(f"Loading dataset '{dataset_name}' (split: {split})...")
        dataset = load_dataset(dataset_name, split=split, **kwargs)
        
        # 샘플 개수 제한
        if num_samples is not None and num_samples < len(dataset):
            dataset = dataset.select(range(num_samples))
            print(f"Selected {num_samples} samples from the dataset")
        else:
            print(f"Using all {len(dataset)} samples from the dataset")
        
        # PyTorch Dataset으로 변환
        pytorch_dataset = HuggingFaceDataset(dataset)
        
        # DataLoader 생성
        dataloader = DataLoader(
            pytorch_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=lambda x: x  # 원본 데이터 그대로 반환
        )
        
        print(f"DataLoader created with {len(dataloader)} batches (batch_size={batch_size})")
        return dataloader
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise


def print_dataset_info(dataloader):
    """DataLoader의 첫 번째 배치 정보를 출력"""
    try:
        first_batch = next(iter(dataloader))
        print(f"\n=== Dataset Info ===")
        print(f"Batch size: {len(first_batch)}")
        print(f"First sample keys: {first_batch[0].keys() if isinstance(first_batch[0], dict) else 'N/A'}")
        print(f"\nFirst sample:")
        print(first_batch[0])
        print("=" * 50)
    except Exception as e:
        print(f"Error printing dataset info: {e}")


# 간단한 사용 예제
if __name__ == "__main__":
    # 예제 1: SQuAD 데이터셋
    print("\n" + "="*50)
    print("Example 1: Loading SQuAD dataset")
    print("="*50)
    try:
        dataloader = get_hf_dataloader(
            dataset_name="squad",
            num_samples=10,
            split="train",
            batch_size=1
        )
        print_dataset_info(dataloader)
    except Exception as e:
        print(f"Failed to load SQuAD: {e}")
    
    # 예제 2: WikiText 데이터셋
    print("\n" + "="*50)
    print("Example 2: Loading WikiText dataset")
    print("="*50)
    try:
        dataloader = get_hf_dataloader(
            dataset_name="wikitext",
            name="wikitext-2-raw-v1",
            num_samples=5,
            split="test",
            batch_size=1
        )
        print_dataset_info(dataloader)
    except Exception as e:
        print(f"Failed to load WikiText: {e}")
        
    # 예제 3: CNN/DailyMail 데이터셋
    print("\n" + "="*50)
    print("Example 3: CNN/DailyMail dataset")
    print("="*50)
    try:
        dataloader = get_hf_dataloader(
            dataset_name="abisee/cnn_dailymail",
            name="3.0.0",
            num_samples=50,
            split="test",
            batch_size=1
        )
        print_dataset_info(dataloader)
    except Exception as e:
        print(f"Failed to load CNN/DailyMail: {e}")
    
