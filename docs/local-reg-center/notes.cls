%-------- Base class load
\NeedsTeXFormat{LaTeX2e}
\LoadClass{article} % Load the base class

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% Used Packages %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\RequirePackage[left=0.7in,right=0.7in,top=1in,bottom=1in]{geometry} 
\RequirePackage[many]{tcolorbox} % Colorbox
\RequirePackage[export]{adjustbox} % Making subsection section
\RequirePackage{amsmath, amssymb, amsthm, mathtools, bm, mathrsfs, nicematrix, 
                arydshln, booktabs, array, bbm, graphicx, xcolor, framed, 
                wrapfig, multicol, url, tikz, enumitem, hyperref, pifont,
                setspace, subcaption}

% % Add Chinese support
\usepackage{fontspec} % For font selection
\usepackage{xeCJK}
\IfFontExistsTF{Kaiti SC}{%
  \setCJKmainfont{Kaiti SC}[AutoFakeBold=2.5]%
  \setCJKmonofont{Kaiti SC}[AutoFakeBold=2.5]%
  \newcommand{\customkaifont}{Kaiti SC}%
}{%
  \setCJKmainfont{FandolKai-Regular.otf}[AutoFakeBold=2.5]%
  \setCJKmonofont{FandolKai-Regular.otf}[AutoFakeBold=2.5]%
  \newcommand{\customkaifont}{FandolKai-Regular.otf}%
}

% Make subsection section
\renewcommand{\thesubsection}{\arabic{subsection}}

% Configure list enumeration
\setlist[enumerate,1]{label={(\roman*)}, itemsep=0.5mm}
\setlist[itemize]{itemsep=0.5mm}

% Configure reference order: unsrt(order), plain(name)
\bibliographystyle{unsrt} 

% Configure the email in title
\newcommand{\tmemail}[1]{\\ \textit{Email:} \texttt{#1}}

% Configure the linespace
\setstretch{1.2}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%% Page Header as Title %%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancypagestyle{firstpage}{ % Define a custom style for the first page
    \fancyhf{}
    \lhead{
      \textsc{Local Regression}
    }
    \rhead{
      \textsc{Student: \href{mailto:xuelin@u.nus.edu}{Zhu Xuelin}}
  }
    \cfoot{\thepage} % Add page number in the center of the footer
    \renewcommand{\headrulewidth}{0.4pt} % Add a horizontal line under the header
    \renewcommand{\footrulewidth}{0pt} % No horizontal line in the footer
}
\fancypagestyle{plain}{ % Define no header for other pages
    \fancyhf{} % Clear all header and footer content
    \cfoot{\thepage} % Add page number in the center of the footer
    \renewcommand{\headrulewidth}{0pt} % No header line
    \renewcommand{\footrulewidth}{0pt} % No footer line
}
\pagestyle{plain} % Default style for subsequent pages
\AtBeginDocument{%
    \fancyheadoffset[L,R]{0pt}%
    \setlength{\headwidth}{\textwidth}%
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%% Custom Symbols %%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcommand{\cmark}{\ding{51}} % Check mark
\newcommand{\xmark}{\ding{55}} % Cross mark
\newcommand*{\circled}[1]{%
    \lower.7ex\hbox{\tikz\draw (0pt, 0pt) circle (.5em) node {\makebox[1em][c]{\small #1}};}%
} % Circled Numbers

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%% Math Shortkeys %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Vectors and matrices
\newcommand{\vecY}{\mathbf{Y}}
\newcommand{\vecX}{\mathbf{X}}
\newcommand{\matY}{\mathbb{Y}}
\newcommand{\matX}{\mathbb{X}}
\newcommand{\obsw}{\mathbf{w}}
\newcommand{\obsy}{\mathbf{y}}
\newcommand{\obsx}{\mathbf{x}}
\newcommand{\vu}{\mathbf{u}}
\newcommand{\vv}{\mathbf{v}}
\newcommand{\va}{\mathbf{a}}
\newcommand{\vzero}{\mathbf{0}}

% Bold symbols
\newcommand{\bb}{\bm{\beta}}
\newcommand{\bmu}{\bm{\mu}}
\newcommand{\balpha}{\bm{\alpha}}
\newcommand{\btheta}{\bm{\theta}}
\newcommand{\bTheta}{\bm{\Theta}}

% Sets and numbers
\newcommand{\bbN}{\mathbb{N}}
\newcommand{\bbZ}{\mathbb{Z}}
\newcommand{\bbR}{\mathbb{R}}
\newcommand{\bbRext}{\overline{\mathbb{R}}}
\newcommand{\bbS}{\mathbb{S}}
\newcommand{\calB}{\mathcal{B}}
\newcommand{\borelR}{\mathcal{B}_\mathbb{R}}

% Mathematical operators
\newcommand{\Var}{\mathrm{Var}}
\newcommand{\Cov}{\mathrm{Cov}}
\newcommand{\E}{\mathbb{E}}
\renewcommand{\P}{\mathbb{P}}
\renewcommand{\Re}{\operatorname{Re}}
\renewcommand{\Im}{\operatorname{Im}}
\renewcommand{\L}{\mathcal{L}}
\newcommand{\ci}{\mathbbm{i}}
\newcommand{\indicator}{\mathbbm{1}}

% Distributions
\newcommand{\Betad}{\mathrm{Beta}}
\newcommand{\Gammad}{\mathrm{Gamma}}
\newcommand{\Multi}{\mathrm{Multi}}
\newcommand{\Ber}{\mathrm{Ber}}
\newcommand{\Bin}{\mathrm{Bin}}
\newcommand{\Pois}{\mathrm{Pois}}
\newcommand{\expd}{\mathrm{Exp}}
\newcommand{\Cate}{\mathrm{Categorical}}
\newcommand{\Unid}{\mathrm{U}}
\newcommand{\normal}{\mathrm{N}}
\newcommand{\mvnorm}{\bm{\mathrm{N}}}

% Logical connectors
\newcommand{\vs}{\quad\text{v.s.}\quad}
\newcommand{\qandq}{\;\;\text{and}\;\;}
\newcommand{\qwithq}{\;\;\text{with}\;\;}
\newcommand{\qwhereq}{\;\;\text{where}\;\;}
\newcommand{\Rightto}{\;\;\Rightarrow\;\;}
\newcommand{\st}{\quad\text{such that}\quad}

% Independence and relations
\newcommand{\indep}{\perp \!\!\! \perp}
\newcommand{\indd}{\mathrel{\overset{\mathrm{ind}}{\sim}}}
\newcommand{\iid}{\mathrel{\overset{\mathrm{iid}}{\sim}}}
\newcommand{\indeq}{\mathrel{\overset{d}{=}}}

% Convergences
\newcommand{\asto}{\mathrel{\overset{a.s.}{\longrightarrow}}}
\newcommand{\inpto}{\mathrel{\overset{p}{\longrightarrow}}}
\newcommand{\indto}{\mathrel{\overset{d}{\longrightarrow}}}
\newcommand{\weakto}{\mathrel{\overset{w}{\longrightarrow}}}
\newcommand{\lto}{\mathrel{\overset{L_1}{\longrightarrow}}}
\newcommand{\ltto}{\mathrel{\overset{L_2}{\longrightarrow}}}

% Inner product and parenthesis
\newcommand{\innerprod}[2]{\left\langle #1, #2 \right\rangle}
\renewcommand{\l}{\left}
\renewcommand{\r}{\right}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%% Conditionings %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Expectations
\NewDocumentCommand{\expectvar}{mm}{#1\IfValueT{#2}{\nonscript\delimsize\vert\nonscript#2}}
\DeclarePairedDelimiterX{\expectarg}[1]{[}{]}{#1}
\DeclarePairedDelimiterX{\expectsarg}[1]{\{}{\}}{#1}
\DeclarePairedDelimiterX{\expectparg}[1]{(}{)}{#1}

\NewDocumentCommand{\expect}{ e{^} s o >{\SplitArgument{1}{\given}}m }{%
  \mathbb{E}%
  \IfValueT{#1}{{\!}^{#1}}%
  \IfBooleanTF{#2}{\expectarg*{\expectvar#4}}{%
    \IfNoValueTF{#3}{\expectarg{\expectvar#4}}{\expectarg[#3]{\expectvar#4}}%
  }%
}
\NewDocumentCommand{\expects}{ e{^} s o >{\SplitArgument{1}{\given}}m }{%
  \mathbb{E}%
  \IfValueT{#1}{{\!}^{#1}}%
  \IfBooleanTF{#2}{\expectsarg*{\expectvar#4}}{%
    \IfNoValueTF{#3}{\expectsarg{\expectvar#4}}{\expectsarg[#3]{\expectvar#4}}%
  }%
}
\NewDocumentCommand{\expectp}{ e{^} s o >{\SplitArgument{1}{\given}}m }{%
  \mathbb{E}%
  \IfValueT{#1}{{\!}^{#1}}%
  \IfBooleanTF{#2}{\expectparg*{\expectvar#4}}{%
    \IfNoValueTF{#3}{\expectparg{\expectvar#4}}{\expectparg[#3]{\expectvar#4}}%
  }%
}

% Probabilities
\NewDocumentCommand{\prob}{ e{^} s o >{\SplitArgument{1}{\given}}m }{%
  \mathbb{P}%
  \IfValueT{#1}{{\!}^{#1}}%
  \IfBooleanTF{#2}{\expectarg*{\expectvar#4}}{%
    \IfNoValueTF{#3}{\expectarg{\expectvar#4}}{\expectarg[#3]{\expectvar#4}}%
  }%
}
\NewDocumentCommand{\probs}{ e{^} s o >{\SplitArgument{1}{\given}}m }{%
  \mathbb{P}%
  \IfValueT{#1}{{\!}^{#1}}%
  \IfBooleanTF{#2}{\expectsarg*{\expectvar#4}}{%
    \IfNoValueTF{#3}{\expectsarg{\expectvar#4}}{\expectsarg[#3]{\expectvar#4}}%
  }%
}
\NewDocumentCommand{\probp}{ e{^} s o >{\SplitArgument{1}{\given}}m }{%
  \mathbb{P}%
  \IfValueT{#1}{{\!}^{#1}}%
  \IfBooleanTF{#2}{\expectparg*{\expectvar#4}}{%
    \IfNoValueTF{#3}{\expectparg{\expectvar#4}}{\expectparg[#3]{\expectvar#4}}%
  }%
}

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%% Math Environments %%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% \definecolor{thm-red1}{rgb}{0.99609375, 0.8671875, 0.87109375} % Colors
% \definecolor{thm-red2}{rgb}{0.98046875, 0, 0.0234375}
% \definecolor{def-brown1}{rgb}{0.953125, 0.9453125, 0.8671875}
% \definecolor{def-brown2}{rgb}{0.74609375, 0.2578125, 0.01171875}
% \definecolor{thm-blue1}{rgb}{0.8828125, 0.98828125, 0.9921875}
% \definecolor{thm-blue2}{rgb}{0, 0.515625, 0.7578125}
% \definecolor{lemma-blue1}{rgb}{0.8828125, 0.98828125, 0.9921875}
% \definecolor{lemma-blue2}{rgb}{0, 0.515625, 0.7578125}
% \definecolor{cor-yellow1}{rgb}{0.99609375, 0.99609375, 0.87109375}
% \definecolor{cor-yellow2}{rgb}{0.99609375, 0.99609375, 0.03515625}
% \definecolor{prop-green1}{rgb}{0.875, 0.99609375, 0.87109375}
% \definecolor{prop-green2}{rgb}{0.05859375, 0.4375, 0}
% \definecolor{prob-brown}{rgb}{0.9921875, 0.953125, 0.87109375}

% % Define the 'theorem' environment first with the box styling
% \newtheorem{theorem}{Theorem} % Start with 'theorem' to define the counter
% \tcolorboxenvironment{theorem}{%
%   boxrule=0pt,
%   boxsep=2pt,
%   colback={thm-red1},
%   enhanced jigsaw,
%   borderline west={2pt}{0pt}{thm-red2},
%   sharp corners,
%   before skip=10pt,
%   after skip=10pt,
%   breakable,
% }

% % General macro for math environments (with shared indexing)
% \newcommand{\createMathEnvironment}[4]{%
%   \newtheorem{#1}[theorem]{#2} % Share the counter of the 'theorem' environment
%   \tcolorboxenvironment{#1}{%
%     boxrule=0pt,
%     boxsep=2pt,
%     colback=#3,
%     enhanced jigsaw,
%     borderline west={2pt}{0pt}{#4},
%     sharp corners,
%     before skip=10pt,
%     after skip=10pt,
%     breakable,
%   }
% }

% % Use the macro to define the other environments, all sharing the same counter
% \createMathEnvironment{proposition}{Proposition}{prop-green1}{prop-green2}
% \createMathEnvironment{corollary}{Corollary}{prop-green1}{prop-green2}
% \createMathEnvironment{lemma}{Lemma}{lemma-blue1}{lemma-blue2}

% % Define the 'definition' environment first with the box styling without italic
% \newcommand{\createNonItalicMathEnvironment}[4]{%
%   \theoremstyle{definition} % Ensure non-italic style
%   \newtheorem{#1}[theorem]{#2} % Share the counter with 'theorem'
%   \tcolorboxenvironment{#1}{%
%     boxrule=0pt,
%     boxsep=2pt,
%     colback=#3,
%     enhanced jigsaw,
%     borderline west={2pt}{0pt}{#4},
%     sharp corners,
%     before skip=10pt,
%     after skip=10pt,
%     breakable,
%   }
% }

% % Create the non-italic environments with the box styling
% \createNonItalicMathEnvironment{definition}{Definition}{def-brown1}{def-brown2}
% \createNonItalicMathEnvironment{algorithm}{Algorithm}{lemma-blue1}{lemma-blue2}
% \createNonItalicMathEnvironment{problem}{Problem}{prob-brown}{prob-brown}
% \createNonItalicMathEnvironment{question}{Question}{prob-brown}{prob-brown}
% \createNonItalicMathEnvironment{assumption}{Assumption}{cor-yellow1}{cor-yellow2}
% \createNonItalicMathEnvironment{setup}{Problem Setup}{def-brown1}{def-brown2}
% \createNonItalicMathEnvironment{example-c}{Example}{lemma-blue1}{lemma-blue2}

% % Other environments
% \newtheorem*{remark}{Remark}
% \newtheorem{example}{Example}
% \newtheorem*{solution}{Solution}

% \newtheorem{innercustomque}{Question} % Self-index questions
% \newenvironment{iquestion}[1]
%   {\renewcommand\theinnercustomque{#1}\innercustomque}
%   {\endinnercustomque}
%  \tcolorboxenvironment{iquestion}{
% 	boxrule=0pt,
% 	boxsep=2pt,
% 	colback={prob-brown},
% 	enhanced jigsaw, 
% 	borderline west={2pt}{0pt}{prob-brown},
% 	sharp corners,
% 	before skip=10pt,
% 	after skip=10pt,
% 	breakable,
% 	}

% % \newtheorem{innercustomque}{Example} % Self-index examples
% % \newenvironment{iexample}[1]
% %   {\renewcommand\theinnercustomque{#1}\innercustomque}
% %   {\endinnercustomque}

% \newtheorem*{continuancex}{Ctd. Question \continuanceref} % Continued questions
% \newenvironment{continuance}[1]
%   {\newcommand\continuanceref{\ref{#1}}\continuancex}
%   {\endcontinuancex}
% \tcolorboxenvironment{continuance}{
% 	boxrule=0pt,
% 	boxsep=2pt,
% 	colback={prob-brown},
% 	enhanced jigsaw, 
% 	borderline west={2pt}{0pt}{prob-brown},
% 	sharp corners,
% 	before skip=10pt,
% 	after skip=10pt,
% 	breakable,
% 	}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%% Math Environments %%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\definecolor{thm-red1}{rgb}{0.99609375, 0.8671875, 0.87109375} % Colors
\definecolor{thm-red2}{rgb}{0.98046875, 0, 0.0234375}
\definecolor{def-brown1}{rgb}{0.953125, 0.9453125, 0.8671875}
\definecolor{def-brown2}{rgb}{0.74609375, 0.2578125, 0.01171875}
\definecolor{thm-blue1}{rgb}{0.8828125, 0.98828125, 0.9921875}
\definecolor{thm-blue2}{rgb}{0, 0.515625, 0.7578125}
\definecolor{lemma-blue1}{rgb}{0.8828125, 0.98828125, 0.9921875}
\definecolor{lemma-blue2}{rgb}{0, 0.515625, 0.7578125}
\definecolor{cor-yellow1}{rgb}{0.99609375, 0.99609375, 0.87109375}
\definecolor{cor-yellow2}{rgb}{0.99609375, 0.99609375, 0.03515625}
\definecolor{prop-green1}{rgb}{0.875, 0.99609375, 0.87109375}
\definecolor{prop-green2}{rgb}{0.05859375, 0.4375, 0}
\definecolor{prob-brown}{rgb}{0.9921875, 0.953125, 0.87109375}

% Define the 'theorem' environment first with the box styling
\newtheorem{theorem}{Theorem} % Start with 'theorem' to define the counter
\tcolorboxenvironment{theorem}{%
  boxrule=0pt,
  boxsep=2pt,
  colback={thm-red1},
  enhanced jigsaw,
  borderline west={2pt}{0pt}{thm-red2},
  sharp corners,
  before skip=10pt,
  after skip=10pt,
  breakable,
}

% General macro for math environments (with shared indexing)
\newcommand{\createMathEnvironment}[4]{%
  \newtheorem{#1}[theorem]{#2} % Share the counter of the 'theorem' environment
  \tcolorboxenvironment{#1}{%
    boxrule=0pt,
    boxsep=2pt,
    colback=#3,
    enhanced jigsaw,
    borderline west={2pt}{0pt}{#4},
    sharp corners,
    before skip=10pt,
    after skip=10pt,
    breakable,
  }
}

% Use the macro to define the other environments, all sharing the same counter
\createMathEnvironment{proposition}{Proposition}{prop-green1}{prop-green2}
\createMathEnvironment{corollary}{Corollary}{prop-green1}{prop-green2}
\createMathEnvironment{lemma}{Lemma}{lemma-blue1}{lemma-blue2}

% Define the 'definition' environment first with the box styling without italic
\newcommand{\createNonItalicMathEnvironment}[4]{%
  \theoremstyle{definition} % Ensure non-italic style
  \newtheorem{#1}[theorem]{#2} % Share the counter with 'theorem'
  \tcolorboxenvironment{#1}{%
    boxrule=0pt,
    boxsep=2pt,
    colback=#3,
    enhanced jigsaw,
    borderline west={2pt}{0pt}{#4},
    sharp corners,
    before skip=10pt,
    after skip=10pt,
    breakable,
  }
}

% Create the non-italic environments with the box styling
\createNonItalicMathEnvironment{definition}{Definition}{def-brown1}{def-brown2}
\createNonItalicMathEnvironment{algorithm}{Algorithm}{lemma-blue1}{lemma-blue2}
\createNonItalicMathEnvironment{problem}{Problem}{prob-brown}{prob-brown}
\createNonItalicMathEnvironment{question}{Question}{prob-brown}{prob-brown}
\createNonItalicMathEnvironment{assumption}{Assumption}{cor-yellow1}{cor-yellow2}
\createNonItalicMathEnvironment{setup}{Problem Setup}{def-brown1}{def-brown2}
\createNonItalicMathEnvironment{example-c}{Example}{lemma-blue1}{lemma-blue2}

% Other environments
\newtheorem*{remark}{Remark}
\newtheorem{example}{Example}
\newtheorem*{solution}{Solution}

\newtheorem{innercustomque}{Question} % Self-index questions
\newenvironment{iquestion}[1]
  {\renewcommand\theinnercustomque{#1}\innercustomque}
  {\endinnercustomque}
 \tcolorboxenvironment{iquestion}{
	boxrule=0pt,
	boxsep=2pt,
	colback={prob-brown},
	enhanced jigsaw, 
	borderline west={2pt}{0pt}{prob-brown},
	sharp corners,
	before skip=10pt,
	after skip=10pt,
	breakable,
	}

% \newtheorem{innercustomque}{Example} % Self-index examples
% \newenvironment{iexample}[1]
%   {\renewcommand\theinnercustomque{#1}\innercustomque}
%   {\endinnercustomque}

\newtheorem*{continuancex}{Ctd. Question \continuanceref} % Continued questions
\newenvironment{continuance}[1]
  {\newcommand\continuanceref{\ref{#1}}\continuancex}
  {\endcontinuancex}
\tcolorboxenvironment{continuance}{
	boxrule=0pt,
	boxsep=2pt,
	colback={prob-brown},
	enhanced jigsaw, 
	borderline west={2pt}{0pt}{prob-brown},
	sharp corners,
	before skip=10pt,
	after skip=10pt,
	breakable,
	}

% Create a macro for creating Chinese environments (theorem, corollary, lemma, proposition)
\newcommand{\createChineseMathEnvironment}[3]{%
  \newtheoremstyle{#1-zh}
    {\topsep} % space above
    {\topsep} % space below
    {\fontspec{\customkaifont}\selectfont} % Prefer macOS fonts, fallback to TeX Live Fandol
    {} % indent
    {\bfseries} % header font
    {.} % punctuation
    {.5em} % space after header
    {}
  
  \theoremstyle{#1-zh}
  \newtheorem{#1-zh}[theorem]{#2} % Share the same counter as 'theorem'
  \tcolorboxenvironment{#1-zh}{%
    boxrule=0pt,
    boxsep=2pt,
    colback={#3},
    enhanced jigsaw,
    borderline west={2pt}{0pt}{#3},
    sharp corners,
    before skip=10pt,
    after skip=10pt,
    breakable,
  }
}

% Create the Chinese versions of the theorem, corollary, lemma, and proposition using the macro
\createChineseMathEnvironment{theorem}{Theorem}{thm-red1}
\createChineseMathEnvironment{corollary}{Corollary}{cor-yellow1}
\createChineseMathEnvironment{lemma}{Lemma}{lemma-blue1}
\createChineseMathEnvironment{proposition}{Proposition}{prop-green1}
