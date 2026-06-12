# Documentação em Português

Esta API oferece geração e edição de imagens usando FLUX.2 e FLUX.1, memória
visual de conversa, execução HTTP tradicional e integração com RunPod
Serverless.

## Por onde começar

1. [Início rápido](quickstart.md): configure o ambiente e gere a primeira imagem.
2. [Referência da API](api.md): consulte rotas, campos, respostas e erros.
3. [Memória de conversa](conversation-memory.md): faça alterações iterativas e volte a revisões antigas.
4. [Modelos](models.md): escolha entre FLUX.2 Dev, FLUX.2 Klein e FLUX.1.
5. [Configuração](configuration.md): referência de variáveis de ambiente.
6. [Deploy e operação](deployment.md): Docker, RunPod, cache R2 e escalabilidade.
7. [Segurança e guardrails](guardrails.md): proteções existentes e recomendações.

## Capacidades

- Texto para imagem.
- Edição usando uma ou até quatro imagens de referência.
- Memória visual persistente por `conversation_id`.
- Histórico e seleção de revisões.
- Saída JSON base64 ou PNG binário.
- FLUX.2 Dev 4-bit como padrão para estudo.
- Compatibilidade com FLUX.2 Klein, FLUX.1 Schnell e FLUX.1 Dev.
- ControlNet Canny quando um modelo FLUX.1 está ativo.
- Cache persistente Hugging Face ou Cloudflare R2.
- API HTTP ou handler RunPod Serverless.

## Limitação importante da memória

O modelo não possui memória interna entre requisições. A aplicação salva a
imagem gerada e a envia novamente como referência na próxima requisição da
conversa. Isso mantém continuidade visual, mas cada nova edição ainda é uma
nova inferência.
