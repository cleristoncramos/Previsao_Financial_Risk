"""
Mantém o app do Streamlit Community Cloud ativo.

O Streamlit Community Cloud coloca em hibernação os apps sem visitas por algum tempo. Uma
requisição HTTP simples não conta como visita: é preciso abrir o app em um navegador. Este
script abre o app com o Chromium (Playwright), clica em "Yes, get this app back up!" se ele
estiver hibernando e aguarda o painel carregar. É executado periodicamente pelo workflow
.github/workflows/keep-alive.yml.

Uso:
    pip install playwright
    playwright install chromium
    python scripts/keep_alive.py [URL]
"""

import os
import sys
import time

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

URL_PADRAO = "https://previsaofinancialrisk.streamlit.app"
SELETOR_ACORDAR = '[data-testid="wakeup-button-viewer"]'
SELETOR_APP = '[data-testid="stApp"]'
TEMPO_MAXIMO_S = 300  # O app hibernando pode levar alguns minutos para reiniciar


def app_carregado(pagina) -> bool:
    """O painel roda dentro de um iframe em *.streamlit.app; procura o app em todos os frames."""
    for frame in pagina.frames:
        try:
            if frame.locator(SELETOR_APP).count() > 0:
                return True
        except PlaywrightError:
            continue  # Frame descartado durante a navegação
    return False


def manter_ativo(url: str) -> bool:
    with sync_playwright() as p:
        navegador = p.chromium.launch()
        pagina = navegador.new_page()
        pagina.goto(url, wait_until="domcontentloaded", timeout=120_000)

        acordou = False
        inicio = time.monotonic()
        while time.monotonic() - inicio < TEMPO_MAXIMO_S:
            if app_carregado(pagina):
                situacao = "reativado" if acordou else "já estava ativo"
                print(f"[OK] App carregado ({situacao}) em {time.monotonic() - inicio:.0f} s: {url}")
                pagina.wait_for_timeout(15_000)  # Mantém a sessão aberta por alguns segundos
                navegador.close()
                return True

            botao = pagina.locator(SELETOR_ACORDAR)
            if not acordou and botao.count() > 0 and botao.first.is_visible():
                print("App em hibernação; clicando em 'Yes, get this app back up!'.")
                botao.first.click()
                acordou = True

            pagina.wait_for_timeout(5_000)

        print(f"[ERRO] O app não carregou em {TEMPO_MAXIMO_S} s: {url}")
        pagina.screenshot(path="keep_alive_falha.png", full_page=True)
        navegador.close()
        return False


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("APP_URL", URL_PADRAO)
    sys.exit(0 if manter_ativo(url) else 1)
