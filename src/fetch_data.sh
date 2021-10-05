#!/usr/bin/bash

set -euo pipefail

kaggle competitions download -c nlp-getting-started -p data/

cd data
unzip nlp-getting-started.zip
rm nlp-getting-started.zip
