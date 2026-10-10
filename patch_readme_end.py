with open('README.md', 'r', encoding='utf-8') as f:
    text = f.read()

start_idx = text.find('**Interpretation:**')
if start_idx != -1:
    end_idx = text.find('## Reproducing the Experiments', start_idx)
    text = text[:start_idx] + text[end_idx:]
    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(text)