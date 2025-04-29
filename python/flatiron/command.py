import subprocess

import click
import lunchbox.theme as lbc
# ------------------------------------------------------------------------------

'''
Command line interface to flatiron library
'''

click.Context.formatter_class = lbc.ThemeFormatter


@click.group()
def main():
    pass


@main.command()
def bash_completion():
    '''
    {white}BASH completion code to be written to a _flatiron completion
    file.{clear}
    '''
    cmd = '_FLATIRON_COMPLETE=bash_source flatiron'
    result = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    result.wait()
    click.echo(result.stdout.read())


@main.command()
def zsh_completion():
    '''
    {white}ZSH completion code to be written to a _flatiron completion
    file.{clear}
    '''
    cmd = '_FLATIRON_COMPLETE=zsh_source flatiron'
    result = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    result.wait()
    click.echo(result.stdout.read())


if __name__ == '__main__':
    main()
