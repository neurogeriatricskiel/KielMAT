## Contributing guide
Thanks for considering contributing to our toolbox! KielMAT is an open-source project and we welcome contributions from anyone to further enhance this project

There are lots of ways to contribute, such as:  
- Use the software, and when you find bugs, tell us about them! We can only fix the bugs we know about.  
- Tell us about parts of the documentation that you find confusing or unclear.  
- Tell us about things you wish KielMAT could do, or things it can do but you wish they were easier.  
- Fix bugs.  
- Implement new features.  
- Improve existing tutorials or write new ones.

To report bugs, request new features, or ask about confusing documentation, it’s usually best to open a [new issue](https://github.com/neurogeriatricskiel/KielMAT/issues/new/choose) on GitHub. For better reproducibility, be sure to include information about your operating system and KielMAT version, and (if applicable) include a reproducible code sample that is as short as possible and ideally uses one of [our example datasets](https://neurogeriatricskiel.github.io/KielMAT/datasets/).

### Overview
In general you'll be working with three different copies of the the KielMAT codebase: the official remote copy at [https://github.com/neurogeriatricskiel/KielMAT](https://github.com/neurogeriatricskiel/KielMAT) (usually called ``upstream``), your remote `fork` of the upstream repository (similar URL, but with your username in place of ``KielMAT``, and usually called ``origin``), and the local copy of the codebase on your computer. The typical contribution process is to:

1. synchronize your local copy with ``upstream``
2. make changes to your local copy
3. `push` your changes to ``origin`` (your remote fork of the upstream)
4. submit a `pull request` from your fork into ``upstream``

### Setting up your local development environment
1. Clone the repository
2. Set up the environment via [uv](https://docs.astral.sh/uv/). (If you don't have uv, you can install it by following the [installation instructions](https://docs.astral.sh/uv/getting-started/installation/).) Then go to the repository directory, and run ``uv sync``. This creates a virtual environment in ``.venv`` and installs KielMAT together with all development dependencies. You can run commands inside this environment by prefixing them with ``uv run`` (e.g. ``uv run pytest``), or by activating the environment directly.
3. Make changes to the code
4. Push to your fork
5. Open a Pull request
