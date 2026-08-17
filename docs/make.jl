using Documenter

const REPO_NAME = "MLJTutorial.jl"
const  REPO = Remotes.GitHub("ablaom", REPO_NAME)

using NotebookManagementTools

makedocs(
    modules=Module[NotebookManagementTools,],
    format = Documenter.HTML(
        collapselevel = 1,
        assets = [
            "assets/favicon.ico",
            asset(
                "https://fonts.googleapis.com/css2?family="*
                    "Lato:ital,wght@0,100;0,300;0,400;0,700;0,900;"*
                    "1,100;1,300;1,400;1,700;1,900&"*
                    "family=Montserrat:ital,wght@0,100..900;1,100..900&display=swap",
                class = :css,
            ),
            asset(
                "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/"*
                    "5.15.4/css/all.min.css",
                class = :css,
            ),
        ],
        size_threshold = 10485760,
        repolink = "https://github.com/ablaom/MLJTutorial.jl",
    ),
    pages=[
        "Home" =>                   "index.md",
        "1\\. Data Representation" => "notebooks/01_data_representation/notebook.md",
        "2\\. Models" =>              "notebooks/02_models/notebook.md",
        "3\\. Pipelines" =>           "notebooks/03_pipelines/notebook.md",
        "4\\. Tuning" =>              "notebooks/04_tuning/notebook.md",
        "5\\. Composition" =>         "notebooks/05_composition/notebook.md",
        "Solutions to Exercises" => "notebooks/99_solution_to_exercises/notebook.md",
        "Lightning Tour" => "notebooks/lightning_tour/notebook.md",
        "NotebookManagementTools.jl" => "notebook_management_tools.md",
    ],
    sitename=REPO_NAME,
    warnonly = [:cross_references, :missing_docs],
    repo = Remotes.GitHub("ablaom", REPO_NAME),
)

deploydocs(
    devbranch="dev", # deployment to gh-pages only happens when this is the target
    push_preview=false,
    repo="github.com/ablaom/$REPO_NAME.git",
)
