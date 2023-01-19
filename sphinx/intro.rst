Introduction
============

A library of custom computer vision models.

See `documentation <https://thenewflesh.github.io/flatiron/>` for details.

Installation
============

Python
~~~~~~

``pip install flatiron``

Docker
~~~~~~

1. Install
   `docker <https://docs.docker.com/v17.09/engine/installation>`
2. Install
   `docker-machine <https://docs.docker.com/machine/install-machine>`
   (if running on macOS or Windows)
3. ``docker pull thenewflesh/flatiron:latest``

Docker For Developers
~~~~~~~~~~~~~~~~~~~~~

1. Install
   `docker <https://docs.docker.com/v17.09/engine/installation>`
2. Install
   `docker-machine <https://docs.docker.com/machine/install-machine>`
   (if running on macOS or Windows)
3. Ensure docker-machine has at least 4 GB of memory allocated to it.
4. ``git clone git@github.com:thenewflesh/flatiron.git``
5. ``cd flatiron``
6. ``chmod +x bin/flatiron``
7. ``bin/flatiron start``

The service should take a few minutes to start up.

Run ``bin/flatiron --help`` for more help on the command line tool.
