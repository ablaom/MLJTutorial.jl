using Test
using NotebookManagementTools

root = joinpath(@__DIR__, "..")

testfile = joinpath(root, "test", "runtests.jl")
srcfile = joinpath(root, "src", "NotebookManagementTools.jl")

@testset "dirs_containing" begin
    @test dirs_containing([testfile,]; root) == [dirname(testfile),]
    @test dirs_containing([srcfile,]; root) == [dirname(srcfile),]
    dirs = dirs_containing([testfile, srcfile, testfile, srcfile]; root)
    @test length(dirs) == 2
    @test Set(dirs) == Set([dirname(srcfile), dirname(testfile)])
end

true
