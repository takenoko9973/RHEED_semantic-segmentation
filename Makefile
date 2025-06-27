.PHONY: install clean

# 環境構築
install:
	pip install -e .

# クリーンアップ
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf *.egg-info
	rm -rf build
	rm -rf dist
