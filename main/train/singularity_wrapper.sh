#!/bin/bash

/usr/local/bin/singularity shell $1 <<EOT
$2 -u ${@:3}
EOT

exit 0
