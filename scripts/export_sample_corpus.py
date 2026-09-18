"""Готовит небольшую выборку справки для публикации в репозитории.

Полный комплект справки Business Studio проприетарный и в репозиторий не кладётся.
Выборка нужна, чтобы проект можно было запустить и посмотреть на демо-данных:
берём страницы, покрывающие все сложные конструкции разметки, вместе с их
иллюстрациями.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bsrag.config import load_settings  # noqa: E402
from bsrag.parsing.dokuwiki import DokuWikiConverter  # noqa: E402

# Страницы подобраны так, чтобы в выборку попали таблицы с объединением строк,
# плагин bslink, блоки кода, врезки, иллюстрации и страницы-оглавления.
SAMPLE_PAGES = [
    "ru/manual/interface/manual_list",
    "ru/manual/shortcuts",
    "ru/manual/filter/filter_element",
    "ru/manual/administration/user_rights",
    "ru/manual/administration/backup",
    "ru/manual/administration/search_for_object_by_id",
    "ru/manual/install/requirements",
    "ru/manual/install/install/used_ports",
    "ru/manual/manage_model/branches/branches_conflicts",
    "ru/manual/manage_model/branches/branches_apply",
    "ru/manual/calculated_properties",
    "ru/manual/mail_delivery",
    "ru/manual/mail_delivery/creating_delivery",
    "ru/manual/export_import/export_import_bs_bs",
    "ru/manual/export_import/customizable_data_exchange",
    "ru/manual/report/optimization_of_report_generation_time",
    "ru/manual/interface/core_features_of_interface/grids",
    "ru/manual/interface/navigation",
    "ru/manual/simulation_fca/data_edit",
    "ru/technical_manual/work_via_odata",
    "ru/technical_manual/work_via_ole",
    "ru/technical_manual/work_via_ole/filter",
    "ru/csdesign/bpmodeling/bpmn_notation",
    "ru/manual/manual",
    "ru/manual/install/activation/online_activation",
]


def main() -> None:
    settings = load_settings()
    pages_dir = settings.resolve(settings.corpus.pages_dir)
    media_dir = settings.resolve(settings.corpus.media_dir)
    target = PROJECT_ROOT / "data" / "sample_corpus"

    shutil.rmtree(target, ignore_errors=True)
    converter = DokuWikiConverter()
    copied = images = 0

    for page_id in SAMPLE_PAGES:
        source = pages_dir / f"{page_id}.txt"
        if not source.exists():
            print(f"пропущено (нет файла): {page_id}")
            continue
        destination = target / "pages" / f"{page_id}.txt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied += 1

        for image in converter.convert(source.read_text(encoding="utf-8", errors="replace")).images:
            image_source = media_dir / image
            if not image_source.exists():
                continue
            image_target = target / "media" / image
            image_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_source, image_target)
            images += 1

    size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file()) / 1024 / 1024
    print(f"Скопировано страниц: {copied}, иллюстраций: {images}, объём: {size:.1f} МБ")
    print(f"Каталог: {target}")


if __name__ == "__main__":
    main()
