#!/usr/bin/env python3
"""Token identifiers of the first prompts of the replication sample, as a backbone's own tokenizer and chat
template produce them, hashed so that two hosts can be compared. No model is loaded and no call is made.

    # the prompts, printed by the collector without a call (wording 1, block 1, the order of its seeded permutation)
    python3 code/replication_probe.py --url http://127.0.0.1:1/v1/chat/completions --model x --wording 1 --blocks 1 \
        --block_size 10 --blind_blocks 2 --arms reaction patient shuffled shuffled_d2 shuffled_d3 --plain_chat \
        --print_prompts 500 --out /tmp/unused > prompts_w1_b1_500.txt

    python3 code/provenance/tokenization_check.py SNAPSHOT_DIR mistral|gemma prompts_w1_b1_500.txt [N]

For the third backbone (mistral) the request is tokenized as the server tokenizes it under --tokenizer_mode
mistral: mistral-common's tokenizer from the snapshot's tekken.json, applied to a chat request of one system
message and one user message. For the second backbone (gemma) the snapshot's chat template is applied with
enable_thinking false and the generation prompt added, as the server does for the requests of the study.

One JSON line is printed: the SHA-256 of the prompts file, the number of prompts and of tokens, the SHA-256
over the JSON-encoded lists of token identifiers in order, the library versions and, for gemma, the SHA-256
of the first rendered prompt. Run on both hosts, the lines must agree in every field but the interpreter.
The outputs of 14 November 2026 are in results/replication/backbones_course.json (tokenization_check).
"""
import hashlib, json, platform, sys
from importlib import metadata

snap, kind, pfile = sys.argv[1:4]; N = int(sys.argv[4]) if len(sys.argv) > 4 else 500
raw = open(pfile, 'rb').read()
items = []
for blk in raw.decode('utf-8').split('\n--- ')[:]:
    if '[system]\n' not in blk: continue
    head, rest = blk.split('\n[system]\n', 1)
    s, u = rest.split('\n[user]\n', 1)
    items.append((s, u.rstrip('\n')))
items = items[:N]
h = hashlib.sha256(); n_tok = 0
out = dict(backbone=kind, prompts_file_sha256=hashlib.sha256(raw).hexdigest(), prompts=len(items))
if kind == 'mistral':
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
    from mistral_common.protocol.instruct.request import ChatCompletionRequest
    from mistral_common.protocol.instruct.messages import SystemMessage, UserMessage
    tok = MistralTokenizer.from_file(snap + '/tekken.json')
    for s, u in items:
        ids = tok.encode_chat_completion(ChatCompletionRequest(messages=[SystemMessage(content=s), UserMessage(content=u)])).tokens
        h.update(json.dumps(ids).encode()); n_tok += len(ids)
    libs = ('mistral_common',)
elif kind == 'gemma':
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(snap)
    for s, u in items:
        o = tok.apply_chat_template([{"role": "system", "content": s}, {"role": "user", "content": u}], tokenize=True,
                                    add_generation_prompt=True, enable_thinking=False)
        ids = o["input_ids"] if isinstance(o, dict) or hasattr(o, "keys") else o
        ids = [int(x) for x in ids]
        h.update(json.dumps(ids).encode()); n_tok += len(ids)
    txt = tok.apply_chat_template([{"role": "system", "content": items[0][0]}, {"role": "user", "content": items[0][1]}], tokenize=False,
                                  add_generation_prompt=True, enable_thinking=False)
    out['rendered_first_sha256'] = hashlib.sha256(txt.encode()).hexdigest()
    libs = ('transformers', 'tokenizers')
else:
    sys.exit('the second argument is mistral or gemma')
out.update(tokens=n_tok, ids_sha256=h.hexdigest(), libraries={l: metadata.version(l) for l in libs}, python=platform.python_version())
print(json.dumps(out))
