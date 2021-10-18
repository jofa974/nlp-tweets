#!/usr/bin/bash

set -euo pipefail

kaggle competitions download -c nlp-getting-started -p data/raw

cd data/raw
unzip nlp-getting-started.zip
rm nlp-getting-started.zip
