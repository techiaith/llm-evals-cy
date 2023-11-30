#/bin/bash
OPENAI_EVALS_DIR='../evals'

for d in evals-cymraeg/* ; do
    cp -rv $d/evals/ ${OPENAI_EVALS_DIR}/evals/registry/
    cp -rv $d/data/ ${OPENAI_EVALS_DIR}/evals/registry/
done
