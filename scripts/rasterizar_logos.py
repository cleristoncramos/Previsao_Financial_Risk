"""
Converte os logotipos dos ativos (assets/ativos/*.svg e *.png) em PNG com fundo
transparente e bordas aparadas, em assets/ativos/png/. Esses arquivos são usados
pelo vídeo de apresentação (o Matplotlib não lê SVG).

A conversão de SVG usa o Chrome ou o Edge instalados, em modo headless (sem janela),
apenas sobre arquivos locais.

Uso (na raiz do projeto):
    python scripts/rasterizar_logos.py
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
ORIGEM = BASE_DIR / "assets" / "ativos"
DESTINO = ORIGEM / "png"
LARGURA_MAXIMA = 480

NAVEGADORES = [
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
]


def localizar_navegador() -> str:
    for caminho in NAVEGADORES:
        if caminho.exists():
            return str(caminho)
    for nome in ("google-chrome", "chromium", "chromium-browser", "msedge"):
        encontrado = shutil.which(nome)
        if encontrado:
            return encontrado
    raise RuntimeError("Chrome ou Edge não encontrado para converter os SVGs.")


def aparar(imagem: Image.Image) -> Image.Image:
    """Remove as bordas transparentes e limita a largura."""
    imagem = imagem.convert("RGBA")
    caixa = imagem.getchannel("A").getbbox()
    if caixa:
        imagem = imagem.crop(caixa)
    if imagem.width > LARGURA_MAXIMA:
        imagem = imagem.resize((LARGURA_MAXIMA, round(imagem.height * LARGURA_MAXIMA / imagem.width)), Image.LANCZOS)
    return imagem


def svg_para_png(navegador: str, svg: Path, saida: Path, pasta_temp: Path) -> None:
    pagina = pasta_temp / f"{svg.stem}.html"
    pagina.write_text(
        "<html><body style='margin:0;background:transparent'>"
        f"<img src='{svg.resolve().as_uri()}' style='width:960px;height:540px;object-fit:contain'>"
        "</body></html>",
        encoding="utf-8",
    )
    captura = pasta_temp / f"{svg.stem}.png"
    subprocess.run(
        [
            navegador, "--headless=new", "--disable-gpu", "--hide-scrollbars",
            "--default-background-color=00000000", "--window-size=960,540",
            f"--screenshot={captura}", pagina.resolve().as_uri(),
        ],
        check=True, capture_output=True, timeout=60,
    )
    aparar(Image.open(captura)).save(saida)


def main():
    DESTINO.mkdir(parents=True, exist_ok=True)
    navegador = localizar_navegador()
    with tempfile.TemporaryDirectory() as pasta:
        for arquivo in sorted(ORIGEM.iterdir()):
            saida = DESTINO / f"{arquivo.stem}.png"
            if arquivo.suffix == ".svg":
                svg_para_png(navegador, arquivo, saida, Path(pasta))
            elif arquivo.suffix == ".png":
                aparar(Image.open(arquivo)).save(saida)
            else:
                continue
            print(f"  {arquivo.name:<12} -> {saida.relative_to(BASE_DIR)} {Image.open(saida).size}")


if __name__ == "__main__":
    main()
