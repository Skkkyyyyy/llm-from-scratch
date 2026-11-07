import torch 

def generate_text_simple(model, idx, max_new_tokens, context_size, eos_id=None, top_k=None, temperature=0.0):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):

        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        #top-k
        if top_k is not None:
            top_vals, _ = torch.topk(logits, top_k) #(batch,k), sorted high->low
            kth_best = top_vals[:,-1:] #(batch,1)
            logits = logits.masked_fill(logits < kth_best, float("-inf")) #-inf -> softmax -> 0

        if temperature > 0: 
            logits = logits/temperature 
            probs = torch.softmax(logits,dim=-1)
            idx_next = torch.multinomial(probs,num_samples=1) #draw a random token, the chance of being picked = its probi 
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        
        if eos_id is not None and (idx_next == eos_id).all():
            break

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx