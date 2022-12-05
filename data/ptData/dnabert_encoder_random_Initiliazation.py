# By Zeheng Bai
"""Random initialization of encoding (non-embedding) layers of DNABERT model"""

from transformers import BertModel, BertConfig
from transformers.tokenization_dna import DNATokenizer

seed = 0
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

config = BertConfig.from_pretrained('transformers/dnabert-config/bert-config-5')
model = BertModel(config=config)
ref_model = BertModel.from_pretrained('5-new-12w-0')
model.embeddings = ref_model.embeddings
model.save_pretrained('empty_dnabert_5_embedding')