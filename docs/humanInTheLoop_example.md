# URSA Human-in-the-Loop Agent Interface Documentation

## Previous Human-in-the-Loop Example has been Depreciated

The HITL interface can now be accessed via the URSA CLI, launch with:

`$ ursa`

and help on commands can he accessed with 

`$ ursa --help`


## Basic Usage

To prompt an URSA agent through the CLI, first select an agent, then issue a prompt to the agent:

```
ursa> execute
execute: Make me a histogram of the first 10000 prime number spacings
```

You can also issue the prompt in one line by prepending the agent name:
```
ursa> execute Make me a histogram of the first 10000 prime number spacings`
```

to see the names of available agents, prompt the CLI with `help`:
```
ursa> help

Documented commands (type help <topic>):
========================================
EOF  agents  arxiv  chat  clear  execute  exit  help  models  web

Undocumented commands:
======================
hypothesize  plan

```


Some additional documentation on the URSA github repo: [LINK](github.com/lanl/ursa)
with more to come.

We should have an in-depth documentation for it, but right now it's documented a bit on the main README and through the help flags with the CLI call. 
