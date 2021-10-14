using MLJTutorial

df = load_horse();
nms = names(df)
@test length(nms) === 16
@test "age" in nms

true
