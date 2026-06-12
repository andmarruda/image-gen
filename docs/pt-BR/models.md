# Modelos: FLUX.2 e FLUX.1

O modelo é escolhido no startup por `MODEL_ID`. Uma instância carrega um modelo
por vez; altere a variável e reinicie para trocar.

## Comparação

| Modelo | Família detectada | Padrão de passos/guidance | Uso sugerido |
|---|---|---|---|
| `black-forest-labs/FLUX.2-dev` | `flux2-dev` | `28` / `4.0` | Estudo, qualidade e edição |
| `black-forest-labs/FLUX.2-klein-4B` | `flux2-klein` | `4` / `1.0` | Menor custo e latência |
| `black-forest-labs/FLUX.1-schnell` | `flux1` | `4` / `0.0` | FLUX.1 rápido |
| `black-forest-labs/FLUX.1-dev` | `flux1` | `28` / `3.5` | FLUX.1 com maior qualidade |

Verifique sempre a licença de cada checkpoint. O FLUX.2 Dev aberto é destinado
a uso não comercial/estudo conforme os termos da BFL.

## FLUX.2 Dev padrão

Configuração recomendada neste projeto:

```env
MODEL_ID=black-forest-labs/FLUX.2-dev
HF_TOKEN=hf_...
FLUX2_DEV_4BIT=true
FLUX2_DEV_QUANTIZED_MODEL_ID=diffusers/FLUX.2-dev-bnb-4bit
FLUX2_TEXT_ENCODER_MODE=remote
MODEL_CPU_OFFLOAD=false
```

Nesse perfil, o runtime carrega o checkpoint quantizado e solicita embeddings
ao encoder remoto configurado em `FLUX2_REMOTE_TEXT_ENCODER_URL`.

Consequências:

- Exige `HF_TOKEN`.
- Depende de rede para o encoder remoto.
- O prompt é enviado ao serviço remoto de encoder.
- O perfil foi escolhido para caber em uma GPU de cerca de 24 GB.

Use `FLUX2_TEXT_ENCODER_MODE=local` somente se o ambiente tiver memória
suficiente para o encoder local.

## FLUX.2 Klein

```env
MODEL_ID=black-forest-labs/FLUX.2-klein-4B
MODEL_CPU_OFFLOAD=false
```

É a opção para reduzir custo e latência. Mantém geração, edição e referências
múltiplas na mesma API.

## FLUX.1

```env
MODEL_ID=black-forest-labs/FLUX.1-schnell
```

ou:

```env
MODEL_ID=black-forest-labs/FLUX.1-dev
HF_TOKEN=hf_...
```

Com FLUX.1:

- Texto para imagem usa `FluxPipeline`.
- Edição usa `FluxImg2ImgPipeline`.
- Apenas a primeira referência é usada em img2img.
- `strength` controla a distância da referência.
- ControlNet Canny fica disponível.

## ControlNet FLUX.1

```env
MODEL_ID=black-forest-labs/FLUX.1-dev
CONTROLNET_MODEL_ID=InstantX/FLUX.1-dev-Controlnet-Canny
DOWNLOAD_CONTROLNET=true
```

Use `POST /generate/controlnet` para preservar bordas/composição. FLUX.2 não usa
essa rota; sua edição nativa com referências acontece em `/generate/edit`.

## FLUX.1/2 como guardrail?

FLUX.1 ou FLUX.2 podem atuar como etapas diferentes de geração, edição ou
validação visual experimental, mas nenhum deles é um guardrail de segurança
confiável. Não use o modelo gerador para decidir sozinho se conteúdo é seguro.

Para guardrails reais, consulte [Segurança e guardrails](guardrails.md).

