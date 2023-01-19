# Introduction
A library of custom computer vision models.

See [documentation](https://thenewflesh.github.io/flatiron/) for details.

# Installation
### Python
`pip install flatiron`

### Docker
1. Install [docker-desktop](https://docs.docker.com/desktop/)
2. `docker pull thenewflesh/flatiron:[version]`

### Docker For Developers
1. Install [docker-desktop](https://docs.docker.com/desktop/)
2. Ensure docker-desktop has at least 4 GB of memory allocated to it.
3. `git clone git@github.com:thenewflesh/flatiron.git`
4. `cd flatiron`
6. `chmod +x bin/flatiron`
7. `bin/flatiron start`

The service should take a few minutes to start up.

Run `bin/flatiron --help` for more help on the command line tool.

---

# Production CLI

flatiron comes with a command line interface defined in command.py.

Its usage pattern is: `flatiron COMMAND [ARGS] [FLAGS] [-h --help]`

## Commands

---

### bash-completion
Prints BASH completion code to be written to a _flatiron completion file

Usage: `flatiron bash-completion`

---

### zsh-completion
Prints ZSH completion code to be written to a _flatiron completion file

Usage: `flatiron zsh-completion`

---

# Development CLI
bin/flatiron is a command line interface (defined in cli.py) that works with
any version of python 2.7 and above, as it has no dependencies.

Its usage pattern is: `bin/flatiron COMMAND [-a --args]=ARGS [-h --help] [--dryrun]`

### Commands

| Command              | Description                                                         |
| -------------------- | ------------------------------------------------------------------- |
| build-package        | Build production version of repo for publishing                     |
| build-prod           | Publish pip package of repo to PyPi                                 |
| build-publish        | Run production tests first then publish pip package of repo to PyPi |
| build-test           | Build test version of repo for prod testing                         |
| docker-build         | Build image of flatiron                                              |
| docker-build-prod    | Build production image of flatiron                                   |
| docker-container     | Display the Docker container id of flatiron                          |
| docker-coverage      | Generate coverage report for flatiron                                |
| docker-destroy       | Shutdown flatiron container and destroy its image                    |
| docker-destroy-prod  | Shutdown flatiron production container and destroy its image         |
| docker-image         | Display the Docker image id of flatiron                              |
| docker-prod          | Start flatiron production container                                  |
| docker-push          | Push flatiron production image to Dockerhub                          |
| docker-remove        | Remove flatiron Docker image                                         |
| docker-restart       | Restart flatiron container                                           |
| docker-start         | Start flatiron container                                             |
| docker-stop          | Stop flatiron container                                              |
| docs                 | Generate sphinx documentation                                       |
| docs-architecture    | Generate architecture.svg diagram from all import statements        |
| docs-full            | Generate documentation, coverage report, diagram and code           |
| docs-metrics         | Generate code metrics report, plots and tables                      |
| library-add          | Add a given package to a given dependency group                     |
| library-graph-dev    | Graph dependencies in dev environment                               |
| library-graph-prod   | Graph dependencies in prod environment                              |
| library-install-dev  | Install all dependencies into dev environment                       |
| library-install-prod | Install all dependencies into prod environment                      |
| library-list-dev     | List packages in dev environment                                    |
| library-list-prod    | List packages in prod environment                                   |
| library-lock-dev     | Resolve dev.lock file                                               |
| library-lock-prod    | Resolve prod.lock file                                              |
| library-remove       | Remove a given package from a given dependency group                |
| library-search       | Search for pip packages                                             |
| library-sync-dev     | Sync dev environment with packages listed in dev.lock               |
| library-sync-prod    | Sync prod environment with packages listed in prod.lock             |
| library-update       | Update dev dependencies                                             |
| session-lab          | Run jupyter lab server                                              |
| session-python       | Run python session with dev dependencies                            |
| state                | State of flatiron                                                    |
| test-coverage        | Generate test coverage report                                       |
| test-dev             | Run all tests                                                       |
| test-fast            | Test all code excepts tests marked with SKIP_SLOWS_TESTS decorator  |
| test-lint            | Run linting and type checking                                       |
| test-prod            | Run tests across all support python versions                        |
| version              | Full resolution of repo: dependencies, linting, tests, docs, etc    |
| version-bump-major   | Bump pyproject major version                                        |
| version-bump-minor   | Bump pyproject minor version                                        |
| version-bump-patch   | Bump pyproject patch version                                        |
| zsh                  | Run ZSH session inside flatiron container                            |
| zsh-complete         | Generate oh-my-zsh completions                                      |
| zsh-root             | Run ZSH session as root inside flatiron container                    |

### Flags

| Short | Long      | Description                                          |
| ----- | --------- | ---------------------------------------------------- |
| -a    | --args    | Additional arguments, this can generally be ignored  |
| -h    | --help    | Prints command help message to stdout                |
|       | --dryrun  | Prints command that would otherwise be run to stdout |
