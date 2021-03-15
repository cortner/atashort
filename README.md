
### Approximation Theory and Applications

This github repository contains the course material for a short-course on
approximation theory taught March-April 2021 at Peking Normal University.

### Installation Instructions

* Download and install `Julia` version 1.5 or 1.6. [[julialang.org]](https://julialang.org)
* Install course material: (1) Git Option: In a terminal clone this repository, then change to the new directory
```
git clone https://github.com/cortner/atashort.git
cd atashort
```
(2) Alternative via zip-file: At the top of this webpage click on [↓ Code] (green button) then [Download Zip]. This will download a file `main.zip` which will contain the latest version of this repository. Unzip it somewhere on your harddrive, open a terminal, change to the directory where the files are and continue as below.
* Start the Julia REPL, and load the dependencies
```
julia --project=. -e "import Pkg; Pkg.resolve(); Pkg.up()"
```
The final lines of the terminal output should look something like this:
```
Status `~/gits/atshort/Project.toml`
  [a93c6f00] DataFrames v0.22.5
  [7a1cc6ca] FFTW v1.3.2
  [b964fa9f] LaTeXStrings v1.2.0
  [91a5bcdd] Plots v1.10.6
  [c3e4b0f8] Pluto v0.12.21
  [08abe8d2] PrettyTables v0.11.1
  [37e2e46d] LinearAlgebra
```
If you see this, then the installation was likely succesful.

Once the installation is complete you can access the course material as
follows:  Change to directory `atashort`, then
```
julia --project=. -e "import Pluto; Pluto.run()"
```
Note that `--project=.` loads the environment specification which tells
Julia which packages to use (the ones we downloaded and installed during the
installation process). Then `import Pluto` loads the `Pluto.jl` package and
`Pluto.run()` starts a webserver on your local machine. It should automatically
open a webbrowser, but if not, then it will print instructions in the terminal,
something like this:
```
(base) Fuji-2:atshort ortner$ j15 --project=. -e "import Pluto; Pluto.run()"

Opening http://localhost:1235/?secret=fKxO12of in your default browser... ~ have fun!

Press Ctrl+C in this terminal to stop Pluto
```
Simply copy-paste the web-address into a browser, then this should load the
Pluto front page. You can now open the sample notebooks to explore Pluto
or open one of the lecture notebooks, e.g. enter `ata_00_intro.jl` into
the text box and click on `Open`.

Some resources for learning about Julia and Pluto:

* https://julialang.org
* https://juliaacademy.com
* https://juliadocs.github.io/Julia-Cheat-Sheet/
* https://github.com/fonsp/Pluto.jl
* https://www.wias-berlin.de/people/fuhrmann/SciComp-WS2021/assets/nb01-first-contact-pluto.html
* https://computationalthinking.mit.edu/Spring21/

Although you won't need it for this course, I recommend VS Code for serious work with Julia. (I still use Atom myself but most development has now moved to VS Code and I will probably follow soon.)
