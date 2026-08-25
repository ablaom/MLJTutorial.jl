using Test
using NotebookManagementTools

root = joinpath(@__DIR__, "dummy_tutorials")
path1 = joinpath(root, "nested", "good_tutorial", "Project.toml")
path2 = joinpath(pwd(), "tutorial_without_scripts", "Project.toml")
path3 = joinpath(root, "bad_tutorial", "Project.toml")

@testset "notebook_dirs_containing" begin
    dirs = notebook_dirs(; root)
    @test dirname(path1) in dirs
    @test !(dirname(path2) in dirs)
    @test length(dirs) == 4
    dirs = notebook_dirs_containing([path1, path2, path3]; root)
    @test dirname(path1) in dirs
    @test !(dirname(path2) in dirs)
    @test dirname(path3) in dirs
    @test length(dirs) == 2
end

true
