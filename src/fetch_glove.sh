#!/usr/bin/bash

set -euo pipefail

mkdir -p ext/
wget https://huggingface.co/stanfordnlp/glove/resolve/main/glove.twitter.27B.zip -P ext/

cd ext/
unzip glove.twitter.27B.zip
rm glove.twitter.27B.zip
