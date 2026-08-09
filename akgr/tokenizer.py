import sys, json

# sys.path.append('./utils/')
from akgr.utils.load_util import load_yaml
# from utils.load_util import load_yaml


from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.processors import TemplateProcessing
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast, T5TokenizerFast, GPT2TokenizerFast
import random

from akgr.reproduction.contracts import (
    ConditionSpec,
    PreparedBatch,
    normalize_condition,
)
def number_to_pattern(input_str):
    elements = input_str.split()

    result = []
    for elem in elements:
        if elem.lstrip('-').isdigit():  # 检查是否是数字（包括负数）
            num = int(elem)
            if num < 0:
                result.append('p')  # 负数变为 p
            else:
                result.append('e')  # 正数变为 e
        else:
            result.append(elem)  # 非数字保持不变
    
    output_str = ' '.join(result)
    return output_str

def number_to_epnumber(input_string):
    elements = input_string.split()  # 按空格分割字符串
    count1 = 0
    count2 = 0
    for elem in elements:
        try:
            num = int(elem)       # 尝试转为浮点数（兼容整数和小数）
            if num > 0:             # 判断是否为正数
                count1 += 1
            elif num < 0:
                count2 += 1
        
        except ValueError:          # 忽略非数字元素（如字母、符号等）
            pass
    count_e = f"{count1}e"
    count_p = f"{count2}p"
    return count_e, count_p

def number_to_epspecific(input_string):
    # 分割字符串并初始化正负集合
    elements = input_string.split()
    positive_numbers = []
    negative_numbers = []
    
    for elem in elements:
        try:
            num = int(elem)
            if num > 0:
                positive_numbers.append(num)
            elif num < 0:
                negative_numbers.append(num)
        except ValueError:
            continue  # 跳过非数字
    
    # # 随机选择（如果集合非空）
    # random_positive = random.choice(positive_numbers) if positive_numbers else None
    # random_negative = random.choice(negative_numbers) if negative_numbers else None
    random_positive = positive_numbers[0] if positive_numbers else None
    random_negative = negative_numbers[0] if negative_numbers else None
    # if 'n' in elements:
    #     random_negative = f"n {random_negative}" if random_negative is not None else None
    #     random_positive = f"n {random_positive}" if random_positive is not None else None
    return str(random_positive), str(random_negative)

def get_vocab(special_tokens, offset, nentity, nrelation):
    vocab = {}
    vocab.update(special_tokens)
    for i in range(1, nentity+1): # [offset, offset + nentity - 1]
        vocab[str(i)] = offset + i - 1
    for i in range(1, nrelation+1): # [offset + nentity, offset + nentity + nrelation - 1]
        vocab[str(-i)] = offset + nentity + i - 1
    # vocab["-1"] = offset
    return vocab, offset + nentity + nrelation

def create_tokenizer(
        special_tokens: dict, offset: int,
        nentity: int, nrelation: int,
        is_gpt: bool):
    pre_tokenizer = WhitespaceSplit()
    vocab, vocab_size = get_vocab(special_tokens, offset=offset, nentity=nentity, nrelation=nrelation)
    model = WordLevel(vocab, unk_token='UNK')
    if not is_gpt:
        post_processor = TemplateProcessing(
            single='$0 END',
            # pair='$A START $B END',
            special_tokens=[('END', special_tokens['END'])]
        )
    else:
        post_processor = TemplateProcessing(
            single='$0 SEP',
            pair='$A SEP $B END',
            special_tokens=[('SEP', special_tokens['SEP']), ('END', special_tokens['END'])]
        )
    tokenizer = Tokenizer(model=model)

    tokenizer.pre_tokenizer = pre_tokenizer
    tokenizer.post_processor = post_processor
    # Just to let the tokenizer know about special tokens
    tokenizer.add_special_tokens(['START', 'END', 'PAD', 'UNK', 'SEP'])
    import io
    from contextlib import redirect_stdout
    trap = io.StringIO()
    with redirect_stdout(trap):
        TokenizerFast = GPT2TokenizerFast if is_gpt else T5TokenizerFast
        tokenizer = TokenizerFast(
            tokenizer_object=tokenizer,
            bos_token='START',
            eos_token='END',
            pad_token='PAD',
            unk_token='UNK',
            sep_token='SEP',
            ) # default padding side
        # tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, vocab_size


def create_reproduction_tokenizer(nentity: int, nrelation: int):
    """Create the strict reproduction tokenizer with a contiguous vocabulary.

    ``create_tokenizer`` intentionally remains unchanged for legacy checkpoints,
    whose numeric IDs contain the historical gap before entity tokens.  Fresh
    reproduction checkpoints use this builder so ``vocab_size == len(tokenizer)``
    and every ID in ``[0, vocab_size)`` is assigned exactly once.
    """
    if nentity <= 0 or nrelation <= 0:
        raise ValueError("nentity and nrelation must be positive")
    ordered_tokens = [
        "PAD", "END", "START", "UNK", "SEP", "COND",
        "(", ")", "e", "p", "i", "u", "n",
        "1p", "2p", "3p", "4p",
        "1e", "2e", "3e", "4e", "5e", "with",
    ]
    ordered_tokens.extend(str(index) for index in range(1, nentity + 1))
    ordered_tokens.extend(str(-index) for index in range(1, nrelation + 1))
    if len(ordered_tokens) != len(set(ordered_tokens)):
        raise ValueError("Reproduction vocabulary contains duplicate tokens")
    vocab = {token: index for index, token in enumerate(ordered_tokens)}
    backend = Tokenizer(WordLevel(vocab, unk_token="UNK"))
    backend.pre_tokenizer = WhitespaceSplit()
    backend.post_processor = TemplateProcessing(
        single="$0 SEP",
        pair="$A SEP $B END",
        special_tokens=[("SEP", vocab["SEP"]), ("END", vocab["END"])],
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="START",
        eos_token="END",
        pad_token="PAD",
        unk_token="UNK",
        sep_token="SEP",
        additional_special_tokens=["COND"],
        padding_side="right",
    )
    actual_ids = sorted(tokenizer.get_vocab().values())
    if actual_ids != list(range(len(tokenizer))):
        raise ValueError("Reproduction tokenizer IDs are not contiguous")
    return tokenizer


def condition_value_from_target(condition: str, target: str) -> str:
    """Return the canonical control value encoded in a conditional prompt."""
    condition = normalize_condition(condition)
    if condition == "unconditional":
        raise ValueError("Unconditional prompts do not have a condition value")
    if condition == "pattern":
        return number_to_pattern(target)
    if condition == "relation_number":
        return number_to_epnumber(target)[1]
    if condition == "entity_number":
        return number_to_epnumber(target)[0]
    if condition == "specific_relation":
        return number_to_epspecific(target)[1]
    if condition == "specific_entity":
        return number_to_epspecific(target)[0]
    raise AssertionError(f"Unhandled normalized condition: {condition}")


def build_prompt(
    source: str,
    condition: ConditionSpec | None,
    tokenizer,
    condition_delimiter: str | None = None,
) -> str:
    """Build a prompt without changing the target-boundary token's meaning.

    Legacy callers default to the historical ``SEP`` delimiter.  Strict
    reproduction configs pass the dedicated ``COND`` token so the sequence is
    ``answers COND condition SEP target END`` in the conditional stage while
    the unconditional contract remains ``answers SEP target END``.
    """
    if condition is None:
        return source
    delimiter = condition_delimiter or tokenizer.sep_token
    if not delimiter:
        raise ValueError("A condition delimiter must be configured")
    if tokenizer.convert_tokens_to_ids(delimiter) == tokenizer.unk_token_id:
        raise ValueError(f"Condition delimiter {delimiter!r} resolves to UNK")
    return f"{source} {delimiter} {condition.value}"


def prepare_batch(
    device,
    sample,
    tokenizer,
    is_gpt: bool,
    src_len: int,
    tgt_len: int,
    is_gen: bool,
    condition: str = "unconditional",
    condition_delimiter: str | None = None,
) -> PreparedBatch:
    """Prepare an unconditional or controlled batch through one code path."""
    kind = normalize_condition(condition)
    source = list(sample["source"])
    target = list(sample["target"])
    pattern_id = sample["pattern_id"]
    conditions = None
    if kind != "unconditional":
        conditions = [ConditionSpec(kind, condition_value_from_target(kind, value)) for value in target]
    prompts = [
        build_prompt(value, spec, tokenizer, condition_delimiter=condition_delimiter)
        for value, spec in zip(source, conditions or [None] * len(source))
    ]

    if not is_gpt:
        prompt_tokens = tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=src_len,
            return_tensors="pt",
        ).to(device)
        input_ids = prompt_tokens.input_ids
        attention_mask = prompt_tokens.attention_mask
        attention_mask[input_ids == tokenizer.eos_token_id] = 0
        labels = tokenizer(
            target,
            padding="max_length",
            truncation=True,
            max_length=tgt_len,
            return_tensors="pt",
        ).input_ids.to(device)
        source_attention_mask = prompt_tokens.attention_mask
    else:
        old_padding_side = tokenizer.padding_side
        try:
            tokenizer.padding_side = "right"
            pair_tokens = tokenizer(
                prompts,
                target,
                padding="longest",
                truncation=True,
                max_length=src_len + tgt_len,
                return_tensors="pt",
            ).to(device)
            labels = pair_tokens.input_ids.clone()
            prompt_tokens_for_mask = tokenizer(
                prompts,
                padding="max_length",
                truncation=True,
                max_length=labels.shape[-1],
                return_tensors="pt",
            ).to(device)
            labels[prompt_tokens_for_mask.attention_mask == 1] = tokenizer.pad_token_id
            if is_gen:
                tokenizer.padding_side = "left"
                prompt_tokens = tokenizer(
                    prompts,
                    padding="longest",
                    truncation=True,
                    max_length=src_len,
                    return_tensors="pt",
                ).to(device)
                input_ids = prompt_tokens.input_ids
                attention_mask = prompt_tokens.attention_mask
                source_attention_mask = prompt_tokens.attention_mask
            else:
                input_ids = pair_tokens.input_ids
                attention_mask = pair_tokens.attention_mask
                source_attention_mask = prompt_tokens_for_mask.attention_mask
        finally:
            tokenizer.padding_side = old_padding_side
            # Fast tokenizers keep the most recent padding direction in the
            # Rust backend.  Clear it so save_pretrained cannot serialize a
            # transient generation-only left-padding state.
            tokenizer.backend_tokenizer.no_padding()

    labels[labels == tokenizer.pad_token_id] = -100
    return PreparedBatch(
        source=source,
        target=target,
        pattern_id=pattern_id,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        source_attention_mask=source_attention_mask,
        conditions=conditions,
    )


def _legacy_prepared_tuple(batch: PreparedBatch):
    base = (
        batch.source,
        batch.target,
        batch.pattern_id,
        batch.input_ids,
        batch.attention_mask,
        batch.labels,
        batch.source_attention_mask,
    )
    if batch.conditions is None:
        return base
    return base + ([condition.value for condition in batch.conditions],)

def search_one_hop(source, graph,src_len):
    G = graph
    new_source_list = []
    for src in source:
            tmp = 0 
            count_list = []
            for node in src.split():
                node = int(node)
                tmp = tmp + 1
                in_edges = G.in_edges(node)  # 获取所有指向 node 的入边
                for (u,v,k) in in_edges:
                    count_list.append(u)
                    
            str_list = set(count_list)
            str_list = list(str_list)
            str_list = map(str, str_list)
            
            new_source = src +' '+ ' '.join(str_list)
            # print(new_source)
            split_list = new_source.split()
            truncated_list = split_list[:src_len]
            truncated_list = ' '.join(truncated_list)
            new_source_list.append(truncated_list)
    return new_source_list
            
           


import torch
def new_extract_sample_to_device(device,
        sample, tokenizer, is_gpt:bool,
        src_len, tgt_len, is_gen:bool):
    return _legacy_prepared_tuple(prepare_batch(
        device, sample, tokenizer, is_gpt, src_len, tgt_len, is_gen, "unconditional"
    ))
    # Legacy implementation retained below for source compatibility/history.
    source = sample['source']
    target = sample['target']
    pattern_id = sample['pattern_id']
    if not is_gpt:
        source_tokenized = tokenizer(
            source,
            padding='max_length',
            max_length=src_len,
            return_tensors="pt").to(device)
        input_ids = source_tokenized.input_ids
        attention_mask = source_tokenized.attention_mask
        # special treatment for T5: ignore end in source
        
        attention_mask[input_ids == tokenizer.eos_token_id] = 0
        

        labels = tokenizer(
            target,
            padding='max_length',
            max_length=tgt_len,
            return_tensors="pt").input_ids.to(device)
    else:
        source_target_tokenized = tokenizer(
            source, target,
            padding='longest',
            # max_length=src_len+tgt_len,
            return_tensors="pt").to(device)
        # labels is the source SEP target END, ...
        labels = torch.clone(source_target_tokenized.input_ids)
        
        # ... with the source part's loss ignored
        source_tokenized = tokenizer(
            source,
            padding='max_length',
            max_length=labels.shape[-1],
            return_tensors="pt").to(device)
        labels[source_tokenized.attention_mask == 1] = tokenizer.pad_token_id
        
        if is_gen == False: # (train/valid) input = source SEP target END, default padding side
            input_ids = source_target_tokenized.input_ids
            attention_mask = source_target_tokenized.attention_mask
            # print('gpt')
            # print(input_ids)
            # print(tokenizer.batch_decode(input_ids))
        else: # (test/optimize) input = source c, left padding (align the last tokens to the right)
            original_padding_side = tokenizer.padding_side
            tokenizer.padding_side = 'left'
            source_tokenized = tokenizer(
                source,
                padding='longest',
                max_length=src_len,
                return_tensors="pt").to(device)
            tokenizer.padding_side = original_padding_side
            input_ids = source_tokenized.input_ids
            attention_mask = source_tokenized.attention_mask

        # labels[source_tokenized.attention_mask == 1] = tokenizer.pad_token_id

    labels[labels == tokenizer.pad_token_id] = -100
    source_attention_mask = source_tokenized.attention_mask

    return source, target, pattern_id, input_ids, attention_mask, labels, source_attention_mask, target

def new_extract_sample_to_device_search(device,
        sample, tokenizer,graph, is_gpt:bool,
        src_len, tgt_len, is_gen:bool):
    source = sample['source']
    target = sample['target']
    pattern_id = sample['pattern_id']
    source = search_one_hop(source, graph, src_len)
    if not is_gpt:
        source_tokenized = tokenizer(
            source,
            padding='max_length',
            max_length=src_len,
            return_tensors="pt").to(device)
        input_ids = source_tokenized.input_ids
        attention_mask = source_tokenized.attention_mask
        # special treatment for T5: ignore end in source
        
        attention_mask[input_ids == tokenizer.eos_token_id] = 0
        

        labels = tokenizer(
            target,
            padding='max_length',
            max_length=tgt_len,
            return_tensors="pt").input_ids.to(device)
    else:
        source_target_tokenized = tokenizer(
            source, target,
            padding='longest',
            # max_length=src_len+tgt_len,
            return_tensors="pt").to(device)
        # labels is the source SEP target END, ...
        labels = torch.clone(source_target_tokenized.input_ids)
        
        # ... with the source part's loss ignored
        source_tokenized = tokenizer(
            source,
            padding='max_length',
            max_length=labels.shape[-1],
            return_tensors="pt").to(device)
        labels[source_tokenized.attention_mask == 1] = tokenizer.pad_token_id
        
        if is_gen == False: # (train/valid) input = source SEP target END, default padding side
            input_ids = source_target_tokenized.input_ids
            attention_mask = source_target_tokenized.attention_mask
            # print('gpt')
            # print(input_ids)
            # print(tokenizer.batch_decode(input_ids))
        else: # (test/optimize) input = source c, left padding (align the last tokens to the right)
            original_padding_side = tokenizer.padding_side
            tokenizer.padding_side = 'left'
            source_tokenized = tokenizer(
                source,
                padding='longest',
                max_length=src_len,
                return_tensors="pt").to(device)
            tokenizer.padding_side = original_padding_side
            input_ids = source_tokenized.input_ids
            attention_mask = source_tokenized.attention_mask

        # labels[source_tokenized.attention_mask == 1] = tokenizer.pad_token_id

    labels[labels == tokenizer.pad_token_id] = -100
    source_attention_mask = source_tokenized.attention_mask

    return source, target, pattern_id, input_ids, attention_mask, labels, source_attention_mask

def new_extract_sample_to_device_pattern(device, sample, tokenizer, is_gpt: bool, src_len, tgt_len, is_gen: bool):
    return _legacy_prepared_tuple(prepare_batch(
        device, sample, tokenizer, is_gpt, src_len, tgt_len, is_gen, "pattern"
    ))
    # Legacy implementation retained below for source compatibility/history.
    source = sample['source']
    target = sample['target']
    pattern_id = sample['pattern_id']
    target_pattern = [number_to_pattern(tgt) for tgt in target]
    # target_pattern = [number_to_epnumber(tgt)[0] for tgt in target]
    merged_source = [f"{s} [SEP] {t}" for s, t in zip(source, target_pattern)]
    # merged_source = f"{source} [SEP] {pattern_id}"  
    # print(merged_source)
    if not is_gpt:
        # 非 GPT 情况：直接处理合并后的 source
        source_tokenized = tokenizer(
            merged_source,
            padding='max_length',
            max_length=src_len,
            return_tensors="pt").to(device)
        input_ids = source_tokenized.input_ids
        attention_mask = source_tokenized.attention_mask
        # 特殊处理 T5：忽略 source 中的 eos_token
        attention_mask[input_ids == tokenizer.eos_token_id] = 0

        labels = tokenizer(
            target,
            padding='max_length',
            max_length=tgt_len,
            return_tensors="pt").input_ids.to(device)
    else:
        # GPT 情况：将合并后的 source 和 target 一起处理
        source_target_tokenized = tokenizer(
            merged_source, target,  # 使用合并后的 source
            padding='longest',
            return_tensors="pt").to(device)
        labels = torch.clone(source_target_tokenized.input_ids)
        
        # 忽略 source 部分的 loss
        source_tokenized = tokenizer(
            merged_source,
            padding='max_length',
            max_length=labels.shape[-1],
            return_tensors="pt").to(device)
        labels[source_tokenized.attention_mask == 1] = tokenizer.pad_token_id
        
        if not is_gen:  # 训练/验证阶段
            input_ids = source_target_tokenized.input_ids
            attention_mask = source_target_tokenized.attention_mask
        else:  # 测试/生成阶段（左填充）
            original_padding_side = tokenizer.padding_side
            tokenizer.padding_side = 'left'
            source_tokenized = tokenizer(
                merged_source,  # 使用合并后的 source
                padding='longest',
                max_length=src_len,
                return_tensors="pt").to(device)
            tokenizer.padding_side = original_padding_side
            input_ids = source_tokenized.input_ids
            attention_mask = source_tokenized.attention_mask

    # 统一处理 labels 的 padding
    labels[labels == tokenizer.pad_token_id] = -100
    source_attention_mask = source_tokenized.attention_mask

    return source, target, pattern_id, input_ids, attention_mask, labels, source_attention_mask, target_pattern

def new_extract_sample_to_device_number_entity(device, sample, tokenizer, is_gpt: bool, src_len, tgt_len, is_gen: bool):
    return _legacy_prepared_tuple(prepare_batch(
        device, sample, tokenizer, is_gpt, src_len, tgt_len, is_gen, "entity_number"
    ))
    # Legacy implementation retained below for source compatibility/history.
    source = sample['source']
    target = sample['target']
    pattern_id = sample['pattern_id']
    target_pattern = [number_to_epnumber(tgt)[0] for tgt in target]
    merged_source = [f"{s} [SEP] {t}" for s, t in zip(source, target_pattern)]
    # merged_source = f"{source} [SEP] {pattern_id}"  
    # print(merged_source)
    if not is_gpt:
        # 非 GPT 情况：直接处理合并后的 source
        source_tokenized = tokenizer(
            merged_source,
            padding='max_length',
            max_length=src_len,
            return_tensors="pt").to(device)
        input_ids = source_tokenized.input_ids
        attention_mask = source_tokenized.attention_mask
        # 特殊处理 T5：忽略 source 中的 eos_token
        attention_mask[input_ids == tokenizer.eos_token_id] = 0

        labels = tokenizer(
            target,
            padding='max_length',
            max_length=tgt_len,
            return_tensors="pt").input_ids.to(device)
    else:
        # GPT 情况：将合并后的 source 和 target 一起处理
        source_target_tokenized = tokenizer(
            merged_source, target,  # 使用合并后的 source
            padding='longest',
            return_tensors="pt").to(device)
        labels = torch.clone(source_target_tokenized.input_ids)
        
        # 忽略 source 部分的 loss
        source_tokenized = tokenizer(
            merged_source,
            padding='max_length',
            max_length=labels.shape[-1],
            return_tensors="pt").to(device)
        labels[source_tokenized.attention_mask == 1] = tokenizer.pad_token_id
        
        if not is_gen:  # 训练/验证阶段
            input_ids = source_target_tokenized.input_ids
            attention_mask = source_target_tokenized.attention_mask
        else:  # 测试/生成阶段（左填充）
            original_padding_side = tokenizer.padding_side
            tokenizer.padding_side = 'left'
            source_tokenized = tokenizer(
                merged_source,  # 使用合并后的 source
                padding='longest',
                max_length=src_len,
                return_tensors="pt").to(device)
            tokenizer.padding_side = original_padding_side
            input_ids = source_tokenized.input_ids
            attention_mask = source_tokenized.attention_mask

    # 统一处理 labels 的 padding
    labels[labels == tokenizer.pad_token_id] = -100
    source_attention_mask = source_tokenized.attention_mask

    return source, target, pattern_id, input_ids, attention_mask, labels, source_attention_mask, target_pattern

def new_extract_sample_to_device_number_relation(device, sample, tokenizer, is_gpt: bool, src_len, tgt_len, is_gen: bool):
    return _legacy_prepared_tuple(prepare_batch(
        device, sample, tokenizer, is_gpt, src_len, tgt_len, is_gen, "relation_number"
    ))
    # Legacy implementation retained below for source compatibility/history.
    source = sample['source']
    target = sample['target']
    pattern_id = sample['pattern_id']
    target_pattern = [number_to_epnumber(tgt)[1] for tgt in target]
    merged_source = [f"{s} [SEP] {t}" for s, t in zip(source, target_pattern)]
    # merged_source = f"{source} [SEP] {pattern_id}"  
    # print(merged_source)
    if not is_gpt:
        # 非 GPT 情况：直接处理合并后的 source
        source_tokenized = tokenizer(
            merged_source,
            padding='max_length',
            max_length=src_len,
            return_tensors="pt").to(device)
        input_ids = source_tokenized.input_ids
        attention_mask = source_tokenized.attention_mask
        # 特殊处理 T5：忽略 source 中的 eos_token
        attention_mask[input_ids == tokenizer.eos_token_id] = 0

        labels = tokenizer(
            target,
            padding='max_length',
            max_length=tgt_len,
            return_tensors="pt").input_ids.to(device)
    else:
        # GPT 情况：将合并后的 source 和 target 一起处理
        source_target_tokenized = tokenizer(
            merged_source, target,  # 使用合并后的 source
            padding='longest',
            return_tensors="pt").to(device)
        labels = torch.clone(source_target_tokenized.input_ids)
        
        # 忽略 source 部分的 loss
        source_tokenized = tokenizer(
            merged_source,
            padding='max_length',
            max_length=labels.shape[-1],
            return_tensors="pt").to(device)
        labels[source_tokenized.attention_mask == 1] = tokenizer.pad_token_id
        
        if not is_gen:  # 训练/验证阶段
            input_ids = source_target_tokenized.input_ids
            attention_mask = source_target_tokenized.attention_mask
        else:  # 测试/生成阶段（左填充）
            original_padding_side = tokenizer.padding_side
            tokenizer.padding_side = 'left'
            source_tokenized = tokenizer(
                merged_source,  # 使用合并后的 source
                padding='longest',
                max_length=src_len,
                return_tensors="pt").to(device)
            tokenizer.padding_side = original_padding_side
            input_ids = source_tokenized.input_ids
            attention_mask = source_tokenized.attention_mask

    # 统一处理 labels 的 padding
    labels[labels == tokenizer.pad_token_id] = -100
    source_attention_mask = source_tokenized.attention_mask

    return source, target, pattern_id, input_ids, attention_mask, labels, source_attention_mask, target_pattern

def new_extract_sample_to_device_specific_relation(device, sample, tokenizer, is_gpt: bool, src_len, tgt_len, is_gen: bool):
    return _legacy_prepared_tuple(prepare_batch(
        device, sample, tokenizer, is_gpt, src_len, tgt_len, is_gen, "specific_relation"
    ))
    # Legacy implementation retained below for source compatibility/history.
    source = sample['source']
    target = sample['target']
    pattern_id = sample['pattern_id']
    target_pattern = [number_to_epspecific(tgt)[1] for tgt in target]
    merged_source = [f"{s} [SEP] {t}" for s, t in zip(source, target_pattern)]
    # merged_source = f"{source} [SEP] {pattern_id}"  
    # print(merged_source)
    if not is_gpt:
        # 非 GPT 情况：直接处理合并后的 source
        source_tokenized = tokenizer(
            merged_source,
            padding='max_length',
            max_length=src_len,
            return_tensors="pt").to(device)
        input_ids = source_tokenized.input_ids
        attention_mask = source_tokenized.attention_mask
        # 特殊处理 T5：忽略 source 中的 eos_token
        attention_mask[input_ids == tokenizer.eos_token_id] = 0

        labels = tokenizer(
            target,
            padding='max_length',
            max_length=tgt_len,
            return_tensors="pt").input_ids.to(device)
    else:
        # GPT 情况：将合并后的 source 和 target 一起处理
        source_target_tokenized = tokenizer(
            merged_source, target,  # 使用合并后的 source
            padding='longest',
            return_tensors="pt").to(device)
        labels = torch.clone(source_target_tokenized.input_ids)
        
        # 忽略 source 部分的 loss
        source_tokenized = tokenizer(
            merged_source,
            padding='max_length',
            max_length=labels.shape[-1],
            return_tensors="pt").to(device)
        labels[source_tokenized.attention_mask == 1] = tokenizer.pad_token_id
        
        if not is_gen:  # 训练/验证阶段
            input_ids = source_target_tokenized.input_ids
            attention_mask = source_target_tokenized.attention_mask
        else:  # 测试/生成阶段（左填充）
            original_padding_side = tokenizer.padding_side
            tokenizer.padding_side = 'left'
            source_tokenized = tokenizer(
                merged_source,  # 使用合并后的 source
                padding='longest',
                max_length=src_len,
                return_tensors="pt").to(device)
            tokenizer.padding_side = original_padding_side
            input_ids = source_tokenized.input_ids
            attention_mask = source_tokenized.attention_mask

    # 统一处理 labels 的 padding
    labels[labels == tokenizer.pad_token_id] = -100
    source_attention_mask = source_tokenized.attention_mask

    return source, target, pattern_id, input_ids, attention_mask, labels, source_attention_mask, target_pattern

def new_extract_sample_to_device_specific_entity(device, sample, tokenizer, is_gpt: bool, src_len, tgt_len, is_gen: bool):
    return _legacy_prepared_tuple(prepare_batch(
        device, sample, tokenizer, is_gpt, src_len, tgt_len, is_gen, "specific_entity"
    ))
    # Legacy implementation retained below for source compatibility/history.
    source = sample['source']
    target = sample['target']
    pattern_id = sample['pattern_id']
    target_pattern = [number_to_epspecific(tgt)[0] for tgt in target]
    merged_source = [f"{s} [SEP] {t}" for s, t in zip(source, target_pattern)]
    # merged_source = f"{source} [SEP] {pattern_id}"  
    # print(merged_source)
    if not is_gpt:
        # 非 GPT 情况：直接处理合并后的 source
        source_tokenized = tokenizer(
            merged_source,
            padding='max_length',
            max_length=src_len,
            return_tensors="pt").to(device)
        input_ids = source_tokenized.input_ids
        attention_mask = source_tokenized.attention_mask
        # 特殊处理 T5：忽略 source 中的 eos_token
        attention_mask[input_ids == tokenizer.eos_token_id] = 0

        labels = tokenizer(
            target,
            padding='max_length',
            max_length=tgt_len,
            return_tensors="pt").input_ids.to(device)
    else:
        # GPT 情况：将合并后的 source 和 target 一起处理
        source_target_tokenized = tokenizer(
            merged_source, target,  # 使用合并后的 source
            padding='longest',
            return_tensors="pt").to(device)
        labels = torch.clone(source_target_tokenized.input_ids)
        
        # 忽略 source 部分的 loss
        source_tokenized = tokenizer(
            merged_source,
            padding='max_length',
            max_length=labels.shape[-1],
            return_tensors="pt").to(device)
        labels[source_tokenized.attention_mask == 1] = tokenizer.pad_token_id
        
        if not is_gen:  # 训练/验证阶段
            input_ids = source_target_tokenized.input_ids
            attention_mask = source_target_tokenized.attention_mask
        else:  # 测试/生成阶段（左填充）
            original_padding_side = tokenizer.padding_side
            tokenizer.padding_side = 'left'
            source_tokenized = tokenizer(
                merged_source,  # 使用合并后的 source
                padding='longest',
                max_length=src_len,
                return_tensors="pt").to(device)
            tokenizer.padding_side = original_padding_side
            input_ids = source_tokenized.input_ids
            attention_mask = source_tokenized.attention_mask

    # 统一处理 labels 的 padding
    labels[labels == tokenizer.pad_token_id] = -100
    source_attention_mask = source_tokenized.attention_mask

    return source, target, pattern_id, input_ids, attention_mask, labels, source_attention_mask, target_pattern

def new_extract_sample_to_device_masked(device,
        sample, tokenizer, is_gpt:bool,
        src_len, tgt_len, is_gen:bool):
    source = sample['source']
    target = sample['target']
    pattern_id = sample['pattern_id']
    if not is_gpt:
        source_tokenized = tokenizer(
            source,
            padding='max_length',
            max_length=src_len,
            return_tensors="pt").to(device)
            
        source_input_ids = source_tokenized.input_ids
        source_attention_mask = source_tokenized.attention_mask
    

        labels = tokenizer(
            target,
            padding='max_length',
            max_length=tgt_len,
            return_tensors="pt").input_ids.to(device)
        labels_input_ids = labels.input_ids
        labels_attention_mask = labels.attention_mask
    else:
        source_target_tokenized = tokenizer(
            source, target,
            padding='longest',
            # max_length=src_len+tgt_len,
            return_tensors="pt").to(device)
        # labels is the source SEP target END, ...
        labels = torch.clone(source_target_tokenized.input_ids)
        # ... with the source part's loss ignored
        source_tokenized = tokenizer(
            source,
            padding='max_length',
            max_length=labels.shape[-1],
            return_tensors="pt").to(device)
        labels[source_tokenized.attention_mask == 1] = tokenizer.pad_token_id

        if is_gen == False: # (train/valid) input = source SEP target END, default padding side
            input_ids = source_target_tokenized.input_ids
            attention_mask = source_target_tokenized.attention_mask
        else: # (test/optimize) input = source SEP, left padding (align the last tokens to the right)
            original_padding_side = tokenizer.padding_side
            tokenizer.padding_side = 'left'
            source_tokenized = tokenizer(
                source,
                padding='longest',
                max_length=src_len,
                return_tensors="pt").to(device)
            tokenizer.padding_side = original_padding_side
            input_ids = source_tokenized.input_ids
            attention_mask = source_tokenized.attention_mask

        # labels[source_tokenized.attention_mask == 1] = tokenizer.pad_token_id
    
    
    input_ids = torch.cat([source_input_ids, labels_input_ids], dim=1)  # 在序列维度上拼接

    # 将 source 和 labels 的 attention_mask 拼接在一起
    combined_attention_mask = torch.cat([source_attention_mask, labels_attention_mask], dim=1)
    labels[labels == tokenizer.pad_token_id] = -100
    source_attention_mask = combined_attention_mask
    attention_mask = source_attention_mask

    return source, target, pattern_id, input_ids, attention_mask, labels, source_attention_mask
def debug():
    config_dataloader = load_yaml('akgr/configs/config-dataloader.yml')
    offset = config_dataloader['offset']
    special_tokens = config_dataloader['special_tokens']
    tokenizer, _ = create_tokenizer(special_tokens, offset, nentity=200000, nrelation=2000, is_gpt=True)
    sample1 = {'answers': [1, 2, 3, 4], "query": ["(","i","(","n","(","p","(",-1,")","(","p","(",0,")","(","e","(",0,")",")",")",")",")","(","p","(",-567,")","(","e","(",24623,")",")",")",")"], "pattern_str":"(i,(n,(p,(p,(e)))),(p,(e)))"}
    sample2 = {'answers': [1, 2], "query": ["(", "p", "(", -1, ")", "(", "e", "(", 0, ")", ")", ")"], "pattern_str": "(p,(e))"}
    from utils.parsing_util import qry_shift_indices, ans_shift_indices, qry_str_2_actionstr
    def list_to_str(l: list) -> str:
        # print('before', l)
        # print('after', ' '.join([str(x) if isinstance(x, int) else x for x in l]))
        return ' '.join([str(x) if isinstance(x, int) else x for x in l])
    sample = {}
    sample['source'] = [list_to_str(ans_shift_indices(sample1['answers'])), list_to_str(ans_shift_indices(sample2['answers']))]
    sample['target'] = [qry_str_2_actionstr(list_to_str(qry_shift_indices(sample1['query']))), qry_str_2_actionstr(list_to_str(qry_shift_indices(sample2['query'])))]
    sample['pattern_id'] = [1, 2]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    source, target, pattern_id, input_ids, attention_mask, labels, source_attention_mask = \
        new_extract_sample_to_device(device, sample, tokenizer, is_gpt=True, src_len=33, tgt_len=66, is_gen=True)
    # input_ids = tokenizer(sample['source'], padding='max_length', max_length=33, return_tensors="pt")
    print('----')
    print('source')
    print(source)
    print('target')
    print(target)
    print('input_ids')
    print(input_ids)
    print('attention_mask')
    print(attention_mask)
    print('labels')
    print(labels)
    labels[labels == -100] = 0
    print(tokenizer.batch_decode(labels, skip_special_tokens=True))


    source, target, pattern_id, input_ids, attention_mask, labels, source_attention_mask = \
        new_extract_sample_to_device(device, sample, tokenizer, is_gpt=True, src_len=33, tgt_len=33, is_gen=False)
    # input_ids = tokenizer(sample['source'], padding='max_length', max_length=33, return_tensors="pt")
    print('source')
    print(source)
    print('target')
    print(target)
    print('input_ids')
    print(input_ids)
    print('attention_mask')
    print(attention_mask)
    print(source_attention_mask)
    print('labels')
    print(labels)
    labels[labels == -100] = 0
    print(tokenizer.batch_decode(labels, skip_special_tokens=True))

def source_to_prompt(sample,args):
    source = sample['source']              # 单个字符串，如 "19346"
    target = sample['target']              # 单个目标
    condition = normalize_condition(args.condition)
    if condition == "unconditional":
        sample["prompt"] = source
        sample["condition"] = None
    else:
        value = condition_value_from_target(condition, target)
        # This legacy mapping helper has no tokenizer argument.  ``SEP`` is the
        # canonical token in create_tokenizer(); callers using another
        # tokenizer should use build_prompt() directly.
        sample["prompt"] = f"{source} SEP {value}"
        sample["condition"] = value
    return sample
if __name__ == '__main__':
    debug()
