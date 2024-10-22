#!/bin/bash

/usr/local/bin/singularity shell $1 <<EOT
cd $2
$3 -u ${@:4}
EOT

exit 0
