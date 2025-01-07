# Role Embeddings

An experimental approach to make language models more robust against many-shot jailbreaks and prompt injections by embedding role information throughout the context. Initial experiments on Llama 3 show promising results in mitigating many-shot jailbreaks more effectively than fine-tuning alone.

## Implementation

3 implementation variants:

- Basic embedding component addition
- Single-layer residual stream intervention
- Multi-layer residual stream intervention
