import torch 

def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch: yours {tuple(left.shape)} vs GPT-2 {tuple(right.shape)}")
    return torch.nn.Parameter(right.clone().detach())
 
 
def load_weights_into_gpt(model, sd, n_layers):
    # embeddings
    model.tok_emb.weight = assign(model.tok_emb.weight, sd["transformer.wte.weight"])
    model.pos_emb.weight = assign(model.pos_emb.weight, sd["transformer.wpe.weight"])
 
    for i in range(n_layers):
        p = f"transformer.h.{i}."
        blk = model.trf_blocks[i]
 
        # attention: Q, K, V stored together in c_attn -> split into 3
        # HF stores weights as (in, out); nn.Linear wants (out, in) -> .T
        q_w, k_w, v_w = torch.chunk(sd[p + "attn.c_attn.weight"], 3, dim=-1)
        q_b, k_b, v_b = torch.chunk(sd[p + "attn.c_attn.bias"], 3, dim=-1)
        blk.attn.W_q.weight = assign(blk.attn.W_q.weight, q_w.T)
        blk.attn.W_k.weight = assign(blk.attn.W_k.weight, k_w.T)
        blk.attn.W_v.weight = assign(blk.attn.W_v.weight, v_w.T)
        blk.attn.W_q.bias = assign(blk.attn.W_q.bias, q_b)
        blk.attn.W_k.bias = assign(blk.attn.W_k.bias, k_b)
        blk.attn.W_v.bias = assign(blk.attn.W_v.bias, v_b)
 
        blk.attn.W_o.weight = assign(blk.attn.W_o.weight, sd[p + "attn.c_proj.weight"].T)
        blk.attn.W_o.bias = assign(blk.attn.W_o.bias, sd[p + "attn.c_proj.bias"])
 
        # feed-forward: layers[0] = expand, layers[2] = project back
        blk.ff.layers[0].weight = assign(blk.ff.layers[0].weight, sd[p + "mlp.c_fc.weight"].T)
        blk.ff.layers[0].bias = assign(blk.ff.layers[0].bias, sd[p + "mlp.c_fc.bias"])
        blk.ff.layers[2].weight = assign(blk.ff.layers[2].weight, sd[p + "mlp.c_proj.weight"].T)
        blk.ff.layers[2].bias = assign(blk.ff.layers[2].bias, sd[p + "mlp.c_proj.bias"])
 
        # layer norms
        blk.norm1.scale = assign(blk.norm1.scale, sd[p + "ln_1.weight"])
        blk.norm1.shift = assign(blk.norm1.shift, sd[p + "ln_1.bias"])
        blk.norm2.scale = assign(blk.norm2.scale, sd[p + "ln_2.weight"])
        blk.norm2.shift = assign(blk.norm2.shift, sd[p + "ln_2.bias"])
 
    model.final_norm.scale = assign(model.final_norm.scale, sd["transformer.ln_f.weight"])
    model.final_norm.shift = assign(model.final_norm.shift, sd["transformer.ln_f.bias"])
 
    # weight tying: output layer reuses the token embedding
    model.out_head.weight = assign(model.out_head.weight, sd["transformer.wte.weight"])
 