import sys
import random

# ============================================================================ #
# output facility

class colors:
    reset          = '\033[0m'
    bold           = '\033[01m'
    disable        = '\033[02m'
    underline      = '\033[04m'
    reverse        = '\033[07m'
    strikethrough  = '\033[09m'
    invisible      = '\033[08m'
    
    class fg:
        black      = '\033[30m'
        red        = '\033[31m'
        green      = '\033[32m'
        orange     = '\033[33m'
        blue       = '\033[34m'
        purple     = '\033[35m'
        cyan       = '\033[36m'
        lightgrey  = '\033[37m'
        darkgrey   = '\033[90m'
        lightred   = '\033[91m'
        lightgreen = '\033[92m'
        yellow     = '\033[93m'
        lightblue  = '\033[94m'
        pink       = '\033[95m'
        lightcyan  = '\033[96m'
    class bg:
        black      = '\033[40m'
        red        = '\033[41m'
        green      = '\033[42m'
        orange     = '\033[43m'
        blue       = '\033[44m'
        purple     = '\033[45m'
        cyan       = '\033[46m'
        lightgrey  = '\033[47m'

# ---------------------------------------------------------------------------- #

def utterWarning (
  text,
  headline = 'Warning',
  indent   = 4,
  headColors = colors.bold + colors.fg.lightred + colors.bg.black,
  textColors = colors.reset
) :
  print(headColors + headline + textColors)
  for line in text.split('\n') :
    print(' ' * indent + line, file = sys.stderr)

# ============================================================================ #

quotes = [
  "GROMACS reminds you: \"I never thought of stopping, and I just hated sleeping. I can't imagine having a better life.\" (Barbara McClintock)",
  "GROMACS reminds you: \"It is disconcerting to reflect on the number of students we have flunked in chemistry for not knowing what we later found to be untrue.\" (Robert L. Weber)",
  "GROMACS reminds you: \"It Just Tastes Better\" (Burger King)",
  "GROMACS reminds you: \"I originally implemented PME to prove that you didn't need it...\" (Erik Lindahl)",
  "GROMACS reminds you: \"I Was Born to Have Adventure\" (F. Zappa)",
  "GROMACS reminds you: \"All You Need is Greed\" (Aztec Camera)",
  "GROMACS reminds you: \"All Work and No Play Makes Jack a Dull Boy\" (The Shining)",
  "GROMACS reminds you: \"By denying scientific principles, one may maintain any paradox.\" (Galileo Galilei)",
  "GROMACS reminds you: \"Get Down In 3D\" (George Clinton)",
  "GROMACS reminds you: \"If Life Seems Jolly Rotten, There's Something You've Forgotten !\" (Monty Python)",
  "GROMACS reminds you: \"I'm a Jerk\" (F. Black)",
  "GROMACS reminds you: \"Why is the Earth moving 'round the sun? Floating in the vacuum with no purpose, not a one\" (Fleet Foxes)",
  "GROMACS reminds you: \"Cut It Deep and Cut It Wide\" (The Walkabouts)",
  "GROMACS reminds you: \"AH ....Satisfaction\" (IRIX imapd)",
  "GROMACS reminds you: \"There was no preconception on what to do\" (Daft Punk)",
  "GROMACS reminds you: \"I have had my results for a long time, but I do not yet know how I am to arrive at them.\" (Carl Friedrich Gauss)",
  "GROMACS reminds you: \"Don't You Wish You Never Met Her, Dirty Blue Gene?\" (Captain Beefheart)",
  "GROMACS reminds you: \"Carbohydrates is all they groove\" (Frank Zappa)",
  "GROMACS reminds you: \"The soul? There's nothing but chemistry here\" (Breaking Bad)",
  "GROMACS reminds you: \"It Doesn't Have to Be Tip Top\" (Pulp Fiction)",
  "GROMACS reminds you: \"Life need not be easy, provided only that it is not empty.\" (Lise Meitner)",
  "GROMACS reminds you: \"It Doesn't Have to Be Tip Top\" (Pulp Fiction)",
  "GROMACS reminds you: \"Breaking the Law, Breaking the Law\" (Judas Priest)",
  "GROMACS reminds you: \"Can I have everything louder than everything else?\" (Deep Purple)",
  "GROMACS reminds you: \"Right Now My Job is Eating These Doughnuts\" (Bodycount)",
  "GROMACS reminds you: \"FORTRAN, the infantile disorder, by now nearly 20 years old, is hopelessly inadequate for whatever computer application you have in mind today: it is now too clumsy, too risky, and too expensive to use.\" (Edsger Dijkstra, 1970)",
  "GROMACS reminds you: \"The only greatness for man is immortality.\" (James Dean)",
  "GROMACS reminds you: \"Try to calculate the numbers that have been\" (The Smoke Fairies)",
  "GROMACS reminds you: \"They don't have half hours in the north\" (Carl Caleman)",
  "GROMACS reminds you: \"You wouldn't walk into a chemistry lab and mix two clear liquids together just because they look pretty much the same, would you?\" (Justin Lemkul)",
  "GROMACS reminds you: \"Carbohydrates is all they groove\" (Frank Zappa)",
  "GROMACS reminds you: \"Any one who considers arithmetical methods of producing random digits is, of course, in a state of sin.\" (John von Neumann)",
  "GROMACS reminds you: \"Hang On to Your Ego\" (F. Black)",
  "GROMACS reminds you: \"I don't know how many of you have ever met Dijkstra, but you probably know that arrogance in computer science is measured in nano-Dijkstras.\" (Alan Kay)",
  "GROMACS reminds you: \"All models are wrong, but some are useful.\" (George Box)",
  "GROMACS reminds you: \"Install our Free Energy Patents app! There is energy all around us; and it's free! Free energy is everywhere, and all around you, just waiting to be extracted! Over 100+ free energy patents!\" (Mind and Miracle Productions on Twitter, spamming a FEP thread)",
  "GROMACS reminds you: \"Sincerity is the key to success. Once you can fake that you've got it made.\" (Groucho Marx)",
  "GROMACS reminds you: \"You could give Aristotle a tutorial. And you could thrill him to the core of his being. Such is the privilege of living after Newton, Darwin, Einstein, Planck, Watson, Crick and their colleagues.\" (Richard Dawkins)",
  "GROMACS reminds you: \"Ramones For Ever\" (P.J. Van Maaren)",
  "GROMACS reminds you: \"We are continually faced by great opportunities brilliantly disguised as insoluble problems.\" (Lee Iacocca)",
  "GROMACS reminds you: \"No, no, you're not thinking, you're just being logical.\" (Niels Bohr)",
  "GROMACS reminds you: \"Check Your Input\" (D. Van Der Spoel)",
  "GROMACS reminds you: \"Insane In Tha Membrane\" (Cypress Hill)",
  "GROMACS reminds you: \"Big Data is like teenage sex: everyone talks about it, nobody really knows how to do it, everyone thinks everyone else is doing it, so everyone claims they are doing it...\" (Dan Ariely)",
  "GROMACS reminds you: \"Let's Unzip And Let's Unfold\" (Red Hot Chili Peppers)",
  "GROMACS reminds you: \"BioBeat is Not Available In Regular Shops\" (P.J. Meulenhoff)",
  "GROMACS reminds you: \"Sitting on a rooftop watching molecules collide\" (A Camp)",
  "GROMACS reminds you: \"I Have a Bad Case Of Purple Diarrhea\" (Urban Dance Squad)",
  "GROMACS reminds you: \"Art is what you can get away with.\" (Andy Warhol)",
  "GROMACS reminds you: \"The World is a Friendly Place\" (Magnapop)",
  "GROMACS reminds you: \"I always seem to get inspiration and renewed vitality by contact with this great novel land of yours which sticks up out of the Atlantic.\" (Winston Churchill)",
  "GROMACS reminds you: \"You Crashed Into the Swamps\" (Silicon Graphics)",
  "GROMACS reminds you: \"I try to identify myself with the atoms... I ask what I would do If I were a carbon atom or a sodium atom.\" (Linus Pauling)",
  "GROMACS reminds you: \"Safety lights are for dudes\" (Ghostbusters 2016)",
  "GROMACS reminds you: \"In ancient times they had no statistics so they had to fall back on lies.\" (Stephen Leacock)",
  "GROMACS reminds you: \"I've basically become a vegetarian since the only meat I'm eating is from animals I've killed myself\" (Mark Zuckerberg)",
  "GROMACS reminds you: \"Sitting on a rooftop watching molecules collide\" (A Camp)",
  "GROMACS reminds you: \"I do not believe continuum electrostatics\" (Arieh Warshel, Nobel lecture 2013)",
  "GROMACS reminds you: \"He's using code that only you and I know\" (Kate Bush)",
  "GROMACS reminds you: \"Take what you want, but just what you need for survival\" (Joe Jackson)",
  "GROMACS reminds you: \"Unfortunately, \"simulation\" has become increasingly misused to mean nothing more than \"calculation\"\" (Bill Jorgensen)"
]

# ASCII art from https://textart.sh/topic/fox
foxes = [
    "                                                                                  \n"
    "                  ████                                        ████                \n"
    "                ▓▓▓▓▓▓▓▓                                    ▓▓▓▓▓▓██              \n"
    "              ██▓▓▓▓▓▓▓▓██                                ██▓▓▓▓▓▓▓▓██            \n"
    "            ██▓▓▓▓▓▓▓▓▓▓██                                ██▓▓▓▓▓▓▓▓▓▓██          \n"
    "            ██▓▓▓▓▓▓▓▓▓▓██                                ██▓▓▓▓▓▓▓▓▓▓██          \n"
    "          ██▓▓▓▓▓▓  ▓▓▓▓▓▓██                            ██▓▓▓▓▓▓░░▓▓▓▓▓▓██        \n"
    "          ██▓▓▓▓▒▒  ▒▒▓▓▓▓██                            ██▓▓▓▓▒▒  ▒▒▓▓▓▓██        \n"
    "          ██▓▓▓▓░░    ▓▓▓▓██                            ██▓▓██      ▓▓▓▓██        \n"
    "          ██▓▓▓▓░░    ▓▓▓▓██                            ██▓▓██      ▓▓▓▓██        \n"
    "        ██▓▓▓▓▓▓░░    ▓▓▓▓▓▓██                        ██▓▓▓▓██      ▓▓▓▓▓▓██      \n"
    "        ██▓▓▓▓░░░░    ░░▓▓▓▓██                        ██▓▓▓▓        ░░▓▓▓▓██      \n"
    "        ██▓▓▓▓░░░░    ░░▓▓▓▓▓▓██                    ██▓▓▓▓▓▓░░    ░░░░▓▓▓▓██      \n"
    "        ██▓▓▓▓░░░░  ░░  ▓▓▓▓▓▓████████████████████████▓▓▓▓▓▓░░░░  ░░░░▓▓▓▓██      \n"
    "        ██▒▒▒▒░░░░░░  ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░▒▒▒▒▒▒██▒▒▒▒▒▒▒▒░░▒▒░░░░▒▒▒▒██      \n"
    "        ██▒▒▒▒░░▒▒  ▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░░░░░░░░░░░░░░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░▒▒░░▒▒▒▒██      \n"
    "        ██▒▒▒▒░░░░▒▒▒▒▒▒▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒▒▒▒▒▒▒▒▒  ░░▒▒▒▒██      \n"
    "    ██████▒▒▒▒  ▒▒▒▒▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒▒▒▒▒▒▒░░▒▒▒▒██████  \n"
    "    ██░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒  ██  \n"
    "    ██  ░░▒▒▒▒▒▒▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒▒▒▒▒▒▒▒▒▒▒░░  ██  \n"
    "    ██  ▒▒▒▒▒▒▒▒▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒▒▒▒▒▒▒▒▒▒▒  ██  \n"
    "      ██  ▒▒▒▒▒▒▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒▒▒▒▒▒▒▒▒  ██    \n"
    "      ██  ▒▒▒▒▒▒▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒▒▒▒▒▒▒▒▒  ██    \n"
    "  ████░░██░░▒▒▒▒▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒▒▒▒▒▒▒  ██  ████\n"
    "  ██░░██░░  ▒▒▒▒▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒▒▒▒▒▒▒  ░░██  ██\n"
    "  ██  ░░██  ▒▒▒▒▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒▒▒▒▒▒▒  ██    ██\n"
    "  ██        ▒▒▒▒▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒▒▒▒▒▒▒        ██\n"
    "    ██      ▒▒▒▒▒▒▒▒░░░░██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░██░░░░▒▒▒▒▒▒▒▒      ██  \n"
    "      ██      ▒▒▒▒▒▒▒▒░░████░░░░░░░░░░░░░░░░░░░░░░░░░░░░████░░▒▒▒▒▒▒▒▒░░    ██    \n"
    "        ██░░    ▒▒▒▒▒▒▒▒██████░░░░░░░░░░░░░░░░░░░░░░░░██████▒▒▒▒▒▒▒▒      ██      \n"
    "    ██████████    ▒▒▒▒▒▒▒▒████░░░░░░░░░░░░░░░░░░░░░░░░████▒▒▒▒▒▒▒▒░░  ██████████  \n"
    "  ██    ░░    ██  ▒▒▒▒▒▒▒▒▒▒████░░░░░░░░░░░░░░░░░░░░████▒▒▒▒▒▒▒▒▒▒  ██        ░░██\n"
    "    ██        ░░    ▒▒▒▒▒▒▒▒▒▒██▒▒▒▒░░░░░░░░░░░░▒▒▒▒██▒▒▒▒▒▒▒▒▒▒    ░░      ░░██  \n"
    "      ██████        ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░░░▒▒▒▒░░░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒        ██████    \n"
    "          ████      ░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░▒▒░░░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒        ▓▓██        \n"
    "            ████████    ▒▒▒▒▒▒▒▒▒▒▒▒░░░░▒▒▒▒░░░░▒▒▒▒▒▒▒▒▒▒▒▒    ████████          \n"
    "                    ████░░▒▒▒▒▒▒▒▒▒▒░░▒▒▒▒▒▒▒▒░░▒▒▒▒▒▒▒▒▒▒░░████                  \n"
    "                        ██  ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒  ██                      \n"
    "                          ██  ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒  ██                        \n"
    "                            ██  ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░██                          \n"
    "                              ██░░▒▒▒▒▓▓▒▒▒▒▓▓▒▒▒▒  ██                            \n"
    "                              ██  ▒▒▒▒▓▓▒▒▓▓▓▓▒▒▒▒  ██                            \n"
    "                                ██░░▒▒▓▓▓▓▓▓▓▓▒▒░░██                              \n"
    "                                ██    ████████    ██                              \n"
    "                                  ▓▓  ████████  ██                                \n"
    "                                  ▒▒▒▒▒▒░░▒▒▒▒██▒▒                                \n"
    "                                    ░░██░░████                                    \n"
    "                                                                                  ",

    "                                        ████                                  \n"
    "                                    ████▒▒██                                  \n"
    "                                  ████  ▒▒██                                  \n"
    "                                ██▒▒  ▒▒▒▒▒▒██                                \n"
    "                              ██▒▒██        ██                                \n"
    "  ████                      ██▒▒██          ██                                \n"
    "██▒▒▒▒██████                ██▒▒██      ▒▒  ████                              \n"
    "██▒▒▒▒██    ████      ██████▒▒▒▒▒▒██    ▒▒▒▒██████████████                    \n"
    "██▒▒    ████▒▒▒▒██████▒▒▒▒▒▒▒▒▒▒▒▒▒▒██▒▒▒▒▒▒██▒▒▒▒▒▒▒▒▒▒▒▒████                \n"
    "██▒▒▒▒      ██▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒██▒▒██▒▒▒▒▒▒▒▒▒▒▒▒▒▒██              \n"
    "  ██▒▒      ██▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒██▒▒██▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒████          \n"
    "  ██        ██▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒██▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒██        \n"
    "  ██▒▒    ██▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒██      \n"
    "  ██▒▒▒▒  ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒  ██▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒██      \n"
    "    ██▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒    ██▒▒▒▒▒▒▒▒▒▒████▒▒▒▒▒▒▒▒██    \n"
    "    ████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒██      ██▒▒▒▒▒▒████▒▒▒▒▒▒▒▒▒▒▒▒██    \n"
    "    ██▒▒██▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒██        ██▒▒▒▒██▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒██    \n"
    "      ██▒▒██▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒██        ██████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒██    \n"
    "      ██▒▒██▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒██      ██▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒██  \n"
    "        ████  ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒    ██▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒██  \n"
    "          ██    ▒▒██████▒▒▒▒▒▒▒▒▒▒▒▒▒▒    ██▒▒  ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒██  \n"
    "          ██            ████▒▒▒▒▒▒▒▒▒▒    ██  ▒▒  ▒▒        ▒▒▒▒▒▒▒▒▒▒▒▒██    \n"
    "            ██                      ██  ████  ▒▒          ▒▒▒▒▒▒▒▒▒▒▒▒▒▒██    \n"
    "              ██                      ██▒▒██              ▒▒  ▒▒▒▒▒▒▒▒▒▒██    \n"
    "                ██████████████████████▒▒▒▒██                    ▒▒▒▒▒▒██      \n"
    "                      ██▒▒      ██▒▒▒▒▒▒▒▒██                    ▒▒▒▒██        \n"
    "                      ██▒▒▒▒  ██▒▒▒▒▒▒▒▒████                  ▒▒▒▒██          \n"
    "                      ██▒▒▒▒▒▒██▒▒▒▒▒▒██  ██                    ██            \n"
    "                        ██████▒▒▒▒▒▒██    ██                ████              \n"
    "                              ██████      ██          ██████                  \n"
    "                                            ██    ████                        \n"
    "                                            ██████                            ",

    "░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░\n"
    "░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░\n"
    "░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒▒▒▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░\n"
    "░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒▒▒▒▒▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░\n"
    "░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒▒▒▒▒▒▒▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░\n"
    "░░░░░░░░░░░░░░░░░░░░░░░░▒▒▒▒▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░\n"
    "░░░░░░░░░░░░░░░░░░░░░░░░▒▒▒▒▒▒▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒▒▒▒▒▓▓▓▓▓▓▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░\n"
    "░░░░░░░░░░░░░░░░░░░░░░░░▒▒▒▒▒▒▒▒▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒▒▒▒▒▓▓▓▓▓▓▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░\n"
    "░░░░░░░░░░░░░░░░░░░░░░░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒▒▒▒▒▒▒▓▓▓▓▓▓▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░\n"
    "░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒▓▓▓▓▒▒▒▒▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒▒▒░░░░░░░░░░░░░░░░░░▒▒▒▒▓▓▓▓▓▓▓▓▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░\n"
    "░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒▓▓▓▓▓▓▓▓▒▒▒▒░░░░░░░░░░░░░░░░▒▒░░░░▒▒░░░░░░░░▒▒░░░░▒▒░░░░░░░░░░░░░░░░▒▒▒▒▓▓████▓▓▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░\n"
    "░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒▒▒▓▓▓▓▓▓▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒▒▒▒▒▒▒▓▓██▓▓▓▓▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░\n"
    "░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒▒▒▓▓▓▓▓▓▓▓▓▓▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒░░▒▒▓▓████▓▓▓▓▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░\n"
    "░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒▒▒▓▓████▓▓▒▒▒▒░░▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░▒▒░░░░░░░░░░░░░░░░▒▒▒▒░░▒▒▓▓████▓▓▓▓▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░\n"
    "░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒▒▒▓▓▓▓██▓▓▓▓▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒▒▒░░░░░░░░░░░░░░░░▒▒▒▒▒▒▓▓██▓▓▓▓▓▓▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░\n"
    "░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒▒▒▓▓▓▓██▓▓▓▓▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒░░░░░░░░░░░░░░░░▒▒▒▒░░▒▒████▓▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░\n"
    "░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒▓▓▓▓▓▓██▓▓▓▓▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒░░░░░░░░░░░░░░░░░░▒▒▒▒▒▒▒▒▓▓▓▓▓▓▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░\n"
    "░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▓▓▓▓▓▓████▓▓░░░░▒▒░░░░░░░░░░░░░░░░░░░░░░▒▒▒▒░░░░░░░░░░░░░░░░▒▒▒▒░░░░▓▓▓▓▒▒▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░\n"
    "░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▓▓▓▓▓▓▓▓▓▓▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒░░░░░░░░░░░░░░░░▒▒▒▒▒▒░░▒▒▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░\n"
    "░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒▒▒▓▓▓▓▒▒░░░░▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒▒▒▒▒▒▒▒▒▓▓▓▓▓▓▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░\n"
    "░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒▒▒▓▓▓▓▓▓▓▓▒▒░░░░▒▒░░░░░░░░░░░░░░░░░░░░░░░░▒▒░░░░░░░░░░░░░░▒▒▒▒▒▒▓▓▒▒▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░\n"
    "░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒▓▓▓▓▒▒▒▒▒▒▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒▒▒▒▒▒▒▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░\n"
    "░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒▒▒▒▒▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒▒▒▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░\n"
    "░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒▒▒▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒▒▒▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░\n"
    "░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒▒▒▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░\n"
    "░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░\n"
    "░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░\n"
    "░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒░░░░░░▒▒▒▒▒▒░░▒▒▒▒▒▒▒▒░░▒▒▒▒▒▒▒▒░░░░▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░\n"
    "░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░░░░░▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░\n"
    "░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░\n"
    "░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒░░░░░░░░░░░░░░▒▒▒▒▒▒▓▓▓▓▓▓▓▓▓▓▓▓▒▒▒▒▒▒▒▒░░▒▒▒▒▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░\n"
    "░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒▒▒▒▒▒▒▓▓▒▒▓▓▓▓▓▓▓▓▒▒▒▒▒▒░░░░▒▒▒▒▒▒▓▓▓▓▓▓▓▓▒▒▓▓▓▓▒▒▒▒▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░\n"
    "░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓▓▓▓▓▓▓▓▒▒▒▒░░░░░░░░▒▒▒▒▓▓▓▓▓▓▓▓░░▓▓▒▒▒▒▒▒▒▒▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░\n"
    "░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒▒▒▒▒░░▒▒▒▒▒▒▓▓▓▓▓▓▓▓▒▒▒▒░░░░░░░░░░▒▒▓▓▓▓▓▓▓▓▒▒▒▒▒▒▒▒░░▒▒▒▒▒▒▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░\n"
    "░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒▒▒▒▒▒▒░░▒▒▒▒▒▒░░░░▒▒▒▒▓▓▒▒░░░░░░░░░░░░▒▒▓▓▒▒▒▒░░▒▒▒▒▒▒▒▒▒▒░░▒▒▒▒▒▒▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░\n"
    "░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒▒▒▒▒▒▒░░░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓▓▒▒░░░░░░░░░░░░▒▒▓▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░░░▒▒▒▒▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░\n"
    "░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒▒▒▒▒▒▒░░░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░░░░░    ░░░░░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░░░▒▒▒▒▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░\n"
    "░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒▒▒▒▒░░░░░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░░░░░    ░░░░░░▒▒▒▒▒▒▒▒░░▒▒▒▒▒▒░░░░░░░░▒▒▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░\n"
    "░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒▒▒▒▒▒▒▒▒░░░░░░░░░░░░░░░░▒▒▒▒▒▒▒▒░░▒▒░░░░░░░░▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░\n"
    "░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒░░░░    ░░▒▒▒▒▒▒▒▒░░░░░░░░  ░░░░░░░░▒▒▒▒▒▒░░░░    ░░▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░\n"
    "░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒▒▒▒▒░░    ░░▒▒▒▒░░░░░░░░      ░░░░░░▒▒▒▒▒▒░░    ░░▒▒▒▒▒▒░░░░░░░░░░      ░░░░░░░░░░░░░░░░░░░░░░░░░░\n"
    "░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░      ░░░░░░░░░░░░▒▒▒▒░░    ░░▒▒▒▒░░░░░░░░░░░░  ░░░░░░░░▒▒▒▒    ░░▒▒▒▒▒▒░░░░░░            ░░░░░░░░░░░░░░░░░░░░░░░░░░\n"
    "░░░░░░░░░░░░░░░░░░░░░░░░░░░░                  ░░░░░░░░░░░░░░▓▓▒▒░░  ░░░░▒▒▒▒▒▒░░░░  ░░▒▒▒▒░░░░░░░░░░░░░░              ░░░░░░░░░░░░░░░░░░░░░░░░░░░░\n"
    "░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░                        ░░░░  ▒▒░░░░  ░░▓▓▒▒▒▒▒▒▓▓░░    ░░░░░░░░░░                      ░░░░░░░░░░░░░░░░░░░░░░░░░░░░\n"
    "░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░    ░░░░░░░░            ░░░░░░      ▒▒▓▓▓▓▓▓▓▓▓▓░░      ▒▒▒▒░░░░            ░░░░░░  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░\n"
    "░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░    ░░░░░░░░░░          ░░▒▒▒▒      ▒▒████▓▓▓▓▓▓░░      ▒▒░░░░            ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░\n"
    "░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒▒▒░░░░    ░░░░░░▒▒░░  ░░░░▓▓██████▒▒░░░░░░▒▒░░░░  ░░    ░░░░▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░\n"
    "░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒▓▓▒▒░░░░      ░░░░░░░░░░░░▒▒▒▒██▒▒▒▒░░░░░░░░░░░░░░░░░░░░▒▒▓▓██▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░\n"
    "░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▓▓▓▓▓▓▒▒░░░░░░░░░░░░░░▒▒▓▓▓▓▓▓▓▓██▓▓▓▓▓▓▓▓▒▒░░░░░░░░░░░░▒▒▓▓██▓▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░\n"
    "░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░████▓▓▓▓▓▓▒▒░░░░░░░░░░░░▒▒▒▒▒▒░░▒▒▒▒▒▒░░▒▒▒▒▒▒░░░░░░░░░░▒▒▓▓▓▓██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░\n"
    "░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒░░░░░░░░░░░░▒▒▒▒▒▒▒▒▒▒░░▒▒▒▒▒▒░░░░░░░░░░▒▒▓▓▓▓▓▓██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░\n"
    "░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▓▓██▓▓██▓▓▓▓▓▓▓▓▒▒░░░░░░░░░░░░░░▒▒▒▒░░░░▒▒░░░░░░░░░░▒▒▒▒▓▓▓▓▓▓████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░\n"
    "░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▓▓▓▓▓▓▓▓▓▓▓▓▒▒████▓▓▒▒▒▒░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░▒▒▒▒▒▒▒▒▓▓▓▓▒▒▓▓▓▓██▓▓░░  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░\n"
    "░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒░░▒▒▓▓██▓▓▓▓▓▓▒▒▒▒▓▓▓▓▒▒▒▒▒▒▒▒▓▓▓▓░░░░▒▒▓▓▓▓▓▓▓▓██▓▓▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░\n"
    "░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▓▓▓▓▓▓██▓▓▓▓▓▓▓▓▒▒▒▒░░░░  ░░▒▒██▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░  ░░▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░\n"
    "░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░██▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒▒▒░░░░░░      ░░████▓▓▒▒      ░░▒▒▒▒▓▓▓▓▓▓▓▓██▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░\n"
    "░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒▒▒░░              ▒▒░░      ░░▒▒▓▓▓▓▓▓▓▓▓▓██▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░\n"
    "░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓██▓▓▓▓▓▓▓▓▒▒▒▒░░      ██████▓▓▓▓░░      ░░▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒▒▒░░░░░░░░░░░░░░░░░░░░\n"
    "░░░░░░░░░░░░░░░░░░░░░░░░░░░░▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓██▓▓▓▓▓▓▓▓▓▓▒▒░░░░  ████▓▓▓▓▓▓▓▓██      ░░▒▒▒▒▓▓▓▓████▓▓▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒▓▓░░░░░░░░░░░░░░░░░░\n"
    "░░░░░░░░░░░░░░░░░░░░░░░░▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓████▒▒▒▒▓▓▓▓▒▒▒▒░░▓▓████▓▓▓▓▓▓▓▓▓▓▓▓      ░░▒▒▓▓▓▓▓▓▓▓▓▓▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒▓▓▓▓░░░░░░░░░░░░░░\n"
    "░░░░░░░░░░░░░░░░░░░░░░▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓████▒▒▒▒▒▒▒▒▒▒▒▒░░████████▓▓▓▓██▒▒▓▓▓▓      ░░▒▒▓▓▓▓▓▓██▓▓▓▓▓▓▓▓▓▓▓▓▒▒▓▓▓▓▓▓▓▓▓▓▒▒▓▓▒▒░░░░░░░░░░░░\n"
    "░░░░░░░░░░░░░░░░░░▒▒▓▓▓▓▓▓▓▓██▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓████▒▒░░▒▒▒▒▒▒░░▓▓▓▓▓▓████▓▓▓▓▓▓░░▒▒▓▓▓▓      ░░▓▓████▓▓▒▒▓▓▓▓▓▓▓▓▓▓▒▒▓▓▓▓▓▓██▓▓▓▓▒▒▓▓░░░░░░░░░░░░\n"
    "░░░░░░░░░░░░░░░░▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓████▒▒░░░░░░▒▒░░▒▒▓▓▓▓▓▓██▓▓▓▓▓▓  ░░▒▒▓▓▒▒    ░░▓▓██▓▓▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒▓▓▓▓▓▓▓▓▓▓▒▒▓▓▓▓░░░░░░░░░░\n"
    "░░░░░░░░░░░░░░▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓██▒▒░░░░░░  ░░░░▒▒▒▒▒▒████████    ░░▓▓▓▓░░  ░░██▓▓▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒▓▓▓▓▓▓▓▓▓▓▓▓▒▒▓▓▓▓░░░░░░░░\n"
    "░░░░░░░░░░░░▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓██░░░░░░▒▒░░░░░░░░▒▒▒▒████▓▓▓▓    ░░░░▒▒▓▓░░▒▒▓▓▓▓▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒▓▓▓▓▓▓▓▓▒▒▓▓▓▓▓▓  ░░░░\n"
    "░░░░░░░░░░▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓████▒▒░░▓▓░░░░░░░░▒▒▒▒████▓▓▓▓    ░░░░▒▒▒▒▓▓▓▓▓▓▓▓▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒▒▒▓▓▓▓▓▓▒▒▓▓▓▓▓▓▓▓▓▓░░░░\n"
    "░░░░░░▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓██▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓██░░░░▓▓░░░░░░░░▒▒▓▓████▓▓▓▓    ░░░░▒▒▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒▒▒▒▒▓▓██▒▒▓▓▓▓▓▓▓▓▓▓▒▒░░\n"
    "░░░░▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░▓▓░░░░░░░░▒▒██████▓▓▓▓    ░░░░▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒▒▒▒▒▓▓▓▓▒▒▓▓▒▒▓▓▓▓▓▓▓▓░░\n"
    "░░░░▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓██████▓▓▒▒▒▒░░░░░░░░▒▒██████▓▓▓▓      ░░▒▒▓▓██▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒▒▒▓▓▓▓▒▒▓▓▒▒▒▒▓▓▓▓▓▓░░\n"
    "░░▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓████▓▓██▒▒▒▒░░░░  ░░▓▓████████▓▓░░    ░░░░████▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒▓▓██▓▓▓▓▒▒▒▒▓▓▒▒▓▓▒▒\n"
    "░░▓▓▓▓████▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓████████░░▒▒░░░░  ░░▓▓██▓▓████▓▓▒▒    ░░░░████▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒▓▓▓▓▓▓▓▓▒▒▒▒▓▓▒▒▓▓▓▓\n"
    "░░██████▓▓▓▓▓▓▓▓▓▓▓▓▓▓██▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓██████▓▓▒▒░░░░  ░░▓▓████████▓▓▒▒    ░░░░████▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒▓▓▓▓▒▒▓▓▓▓▓▓▓▓▒▒▓▓▒▒▓▓▓▓▓▓\n"
    "▒▒██████▓▓▓▓▓▓▓▓▒▒▓▓▓▓██▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓████████▒▒░░░░  ░░██████████▓▓▓▓    ░░  ▓▓██▒▒▓▓▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒▒▒▓▓▒▒▓▓▓▓▓▓▓▓▒▒▓▓▒▒▓▓▓▓▒▒"
]

if len(sys.argv) == 1 :
    utterWarning(
      "No Input Parameters were specified\n"
      "Please start GROMOCKS with a command line argument.",
      "Error"
    )
    sys.exit(1)

print("Please choose one of the following Analysis Parameters:")
print("#  1  LJ-(SR)          2  Disper.-corr.    3  Coulomb-(SR)     4  Coul.-recip.  ")
print("#  5  Potential        6  Kinetic-En.      7  Total-Energy     8  Conserved-En. ")
print("#  9  Temperature     10  Pres.-DC        11  Pressure        12  Box-X         ")
print("# 13  Box-Y           14  Box-Z           15  Volume          16  Density       ")
print("# 17  pV              18  Enthalpy        19  Vir-XX          20  Vir-XY        ")
print("# 21  Vir-XZ          22  Vir-YX          23  Vir-YY          24  Vir-YZ        ")
print("# 25  Vir-ZX          26  Vir-ZY          27  Vir-ZZ          28  Pres-XX       ")
print("# 29  Pres-XY         30  Pres-XZ         31  Pres-YX         32  Pres-YY       ")
print("# 33  Pres-YZ         34  Pres-ZX         35  Pres-ZY         36  Pres-ZZ       ")
print("# 37  #Surf*SurfTen   38  T-System        39  Lamb-System                       ")

selection = input("> ")

if not selection in (f"{i}" for i in range(1, 40) ) :
    utterWarning(
      "Invalid Analysis ID: " + selection,
      "Error"
    )
    sys.exit(1)

print(selection)

foxID = int(selection) % 3

print(foxes[foxID])
print( random.choice(quotes) )