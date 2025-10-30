#!/bin/bash

# Usage: ./download_goes16rrqpe.sh <datetimes.txt>
# Each line in <datetimes.txt> should be in the format: YYYY-MM-DD HH:mm:ss

if [ $# -ne 1 ]; then
  echo "Usage: $0 <datetimes.txt>"
  exit 1
fi

DATETIME_FILE="$1"
PRODUCTS="ABI-L2-RRQPEF"
BUCKET="noaa-goes16"

while IFS= read -r datetime; do
  for offset in 0 1; do
    dt=$(date -d "${datetime} -${offset} hour" +"%Y-%m-%dT%H:%M")
    YYYY=${datetime:0:4}
    MM=${datetime:5:2}
    DD=${datetime:8:2}
    HH=${datetime:11:2}
    DOY=$(date -d "${YYYY}-${MM}-${DD}" +%j)

    for PRODUCT in $PRODUCTS; do
      PREFIX="$PRODUCT/${YYYY}/${DOY}/${HH}"
      aws s3 sync "s3://${BUCKET}/${PREFIX}" "./data/raw/satellite/${PREFIX}" --no-sign-request
    done
  done
done <"$DATETIME_FILE"
