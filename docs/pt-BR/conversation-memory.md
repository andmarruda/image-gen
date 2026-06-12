# Memória de Conversa

## Como funciona

FLUX não mantém estado interno entre chamadas. A API implementa memória visual:

1. Gera uma imagem.
2. Salva `latest.png`, uma revisão PNG e um `manifest.json`.
3. Retorna `conversation_id` e `revision_id`.
4. Na próxima chamada com o mesmo `conversation_id`, carrega a imagem anterior.
5. Envia essa imagem ao modelo como referência junto da nova instrução.

## Criar e continuar uma conversa

Primeira geração:

```bash
curl -X POST "$BASE_URL/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"um robô jardineiro amarelo em uma estufa"}'
```

Continuação:

```bash
curl -X POST "$BASE_URL/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_id":"ID_RETORNADO",
    "prompt":"mantenha o robô e a composição, mas transforme a cena em amanhecer"
  }'
```

Para melhores resultados, descreva explicitamente o que deve permanecer e o
que deve mudar. A memória fornece a imagem; ela não interpreta intenções vagas.

## Voltar a uma revisão antiga

Consulte o histórico:

```bash
curl "$BASE_URL/conversations/ID_RETORNADO"
```

Depois use um `revision_id`:

```json
{
  "conversation_id": "ID_RETORNADO",
  "revision_id": "REVISAO_ANTIGA",
  "prompt": "crie outra direção visual a partir desta versão"
}
```

## Controles

- `use_previous=false`: não usa a última imagem, mas salva o novo resultado na mesma conversa.
- `remember=false`: não recupera nem salva memória para essa chamada.
- Imagem explícita em `image`/`images`: substitui a referência automática.
- `CONVERSATION_MAX_REVISIONS`: limita revisões armazenadas; antigas são apagadas.
- `CONVERSATION_MEMORY_ENABLED=false`: desativa a memória globalmente.

## Persistência e escala

O padrão é `/data/conversations`. Em Docker Compose existe um volume persistente.

Em RunPod Serverless ou múltiplas réplicas, use armazenamento compartilhado.
Sem Network Volume, um worker pode não encontrar a conversa criada por outro e
as conversas desaparecem em cold starts.

O armazenamento atual usa arquivos locais e um lock por processo. Ele é simples
e adequado para estudo. Para alta concorrência distribuída, considere:

- Object storage para PNGs.
- Banco de dados para manifestos.
- Lock distribuído ou filas por conversa.
- Identidade do usuário vinculada ao `conversation_id`.

## Privacidade

As imagens e prompts ficam armazenados até exclusão ou limite de revisões.
Apague uma conversa quando terminar:

```bash
curl -X DELETE "$BASE_URL/conversations/ID_RETORNADO"
```

