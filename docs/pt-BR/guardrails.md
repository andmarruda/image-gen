# Segurança e Guardrails

## Proteções já implementadas

| Proteção | Comportamento |
|---|---|
| API key opcional | Protege todas as rotas, exceto `/health` |
| Comparação segura | Usa comparação constante para a chave |
| Limite de payload | `MAX_REQUEST_MB`, padrão 32 MB |
| Limite de resolução | 256–2048, divisível por 16 |
| Limite de passos | 1–100 |
| Limite de guidance | 0–20 |
| Limite de referências | Máximo de 4 |
| IDs validados | Impede traversal via `conversation_id`/`revision_id` |
| Histórico limitado | `CONVERSATION_MAX_REVISIONS` |
| Inferência serializada | Evita concorrência perigosa na mesma GPU |
| Exclusão de conversa | Remove prompts e imagens armazenadas |

Configure autenticação:

```env
API_KEY=gere-uma-chave-longa-e-aleatoria
MAX_REQUEST_MB=32
```

## O que não está implementado

- Moderação de prompt.
- Moderação da imagem enviada ou gerada.
- Rate limiting por usuário/IP.
- Cotas de GPU ou orçamento.
- Autorização por dono da conversa.
- Criptografia de imagens no armazenamento.
- Expiração automática de conversas.
- Auditoria estruturada.
- Filtro contra prompt injection em sistemas externos.

Não exponha a API diretamente na internet supondo que `API_KEY` resolve todos
esses pontos.

## Arquitetura recomendada de guardrails

```text
Cliente
  -> Gateway com autenticação, rate limit e limite de custo
  -> Moderação do prompt e imagens de entrada
  -> Fila de geração
  -> Worker FLUX
  -> Moderação da imagem gerada
  -> Armazenamento/entrega
```

## Usar FLUX.1 ou FLUX.2 como guardrail

Um modelo gerador pode ajudar experimentalmente a:

- Normalizar ou reconstruir uma imagem.
- Gerar versões com estilo controlado.
- Comparar consistência visual.

Mas FLUX.1/2 não são classificadores de segurança e podem falhar silenciosamente.
Use um modelo/serviço de moderação dedicado antes e depois da geração.

## Proteção de memória

Atualmente, quem conhece um `conversation_id` e possui acesso à API pode
consultar ou excluir aquela conversa. Em ambiente multiusuário:

- Associe conversa ao usuário autenticado em banco de dados.
- Autorize `GET` e `DELETE` por proprietário.
- Use IDs imprevisíveis.
- Defina retenção e expiração.
- Evite guardar dados sensíveis em prompts/imagens.

## Custos e abuso

Limites técnicos reduzem risco, mas não controlam orçamento. Recomendações:

- Rate limit no gateway.
- Máximo de jobs simultâneos por usuário.
- Cota diária de imagens ou segundos de GPU.
- Resoluções e passos permitidos por plano.
- Timeout e cancelamento de jobs.
- Alertas de custo e utilização.

