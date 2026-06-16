import Rouge

# 生成文本
generated_text = "This is some generated text."

# 参考文本列表
reference_texts = ["This is a reference text.", "This is another generated reference text."]

#reference_texts = ["It is a nice day today", "The weather is lovely this afternoon"]

#reference_texts = ["It is a nice day today", "It is national day today"]

# 计算 ROUGE 指标
rouge = Rouge()
scores = rouge.get_scores(generated_text, reference_texts[1])

# 打印结果
print("ROUGE-1 precision:", scores[0]["rouge-1"]["p"])
print("ROUGE-1 recall:", scores[0]["rouge-1"]["r"])
print("ROUGE-1 F1 score:", scores[0]["rouge-1"]["f"])