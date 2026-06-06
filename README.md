# Reproducao de segmentacao em colonoscopia

Reproducao do artigo "Application of Deep Learning Models for Semantic Segmentation of Colonoscopy Images".

## Uso

```bash
make setup
make train MODEL=cascade
```

Modelos em `upstream/commands.yaml`.

Para preparar apenas um modelo upstream, passe `MODEL` no setup:

```bash
make setup MODEL=esfpnet
```

Ambientes Conda que ja existem sao pulados por padrao. Para remover e criar
de novo o ambiente de um modelo, use:

```bash
make setup MODEL=esfpnet RECREATE_ENVS=1
```

Para tentar atualizar um ambiente existente sem apagar antes, use:

```bash
make setup MODEL=esfpnet UPDATE_ENVS=1
```

O setup usa o solver `libmamba` do Conda por padrao. Para trocar o solver:

```bash
make setup MODEL=esfpnet CONDA_SOLVER=classic
```

## Gerar a Tabela 2

A tabela e calculada a partir das mascaras preditas no split de teste. O formato esperado para qualquer modelo e:

```text
outputs/predictions/<modelo>/<stem_da_imagem>.png
```

Depois do treino, exporte as predicoes do modelo desejado:

```bash
make predict MODEL=esfpnet
make predict MODEL=cascade
```

Cada exportador salva as mascaras em `outputs/predictions/<modelo>/`. Depois gere a tabela:

```bash
make table2
```

Os arquivos sao salvos em `outputs/tables/table2.csv` e `outputs/tables/table2.md`.
