"""CLI interface for the Personal Exploration Engine."""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt
from rich.live import Live
from rich.markdown import Markdown

from .models.config import Settings
from .core.orchestrator import ExplorationEngine
from .tools.import_knowledge_graphs import KnowledgeGraphImporter

# Initialize Typer app and Rich console
app = typer.Typer(
    name="peengine",
    help="Personal Exploration Engine - A terminal-based interactive learning tool"
)
console = Console()

# Global engine instance
engine: Optional[ExplorationEngine] = None


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("peengine.log"),
            logging.StreamHandler()
        ]
    )


async def initialize_engine(verbose: bool = False) -> ExplorationEngine:
    """Initialize the exploration engine."""
    global engine
    if engine is None:
        try:
            settings = Settings()
            # Only setup logging if verbose mode is enabled
            if verbose:
                setup_logging(settings.log_level)
            else:
                # Set logging to ERROR level to suppress most logs
                setup_logging("ERROR")
            engine = ExplorationEngine(settings)
            await engine.initialize()
            console.print("[green]✓[/green] Exploration Engine initialized")
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to initialize engine: {e}")
            raise typer.Exit(1)
    return engine


@app.command()
def start(
    topic: str = typer.Argument(..., help="Topic to explore"),
    title: Optional[str] = typer.Option(None, "--title", "-t", help="Session title"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging")
):
    """Start a new exploration session."""
    async def _start_session():
        engine = await initialize_engine(verbose)
        
        try:
            session = await engine.start_session(topic, title)
            console.print(f"[green]✓[/green] Started session: {session.title}")
            console.print(f"Session ID: {session.id}")
            console.print(f"Topic: {session.topic}")
            
            # Enter interactive mode
            await interactive_session(engine)
            
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to start session: {e}")
            raise typer.Exit(1)
    
    asyncio.run(_start_session())


async def interactive_session(engine: ExplorationEngine):
    """Run interactive exploration session."""
    console.print(Panel(
        Markdown("""
# Welcome to Your Exploration Session

**Available commands:**
- `/map` - Show session concept map
- `/gapcheck` - Check understanding gaps  
- `/seed` - Get exploration seed from MA
- `/end` - End current session
- `/help` - Show this help

**Just start exploring by sharing your thoughts or asking questions!**
        """),
        title="Session Started",
        border_style="green"
    ))
    
    while True:
        try:
            # Get user input
            user_input = Prompt.ask("[bold cyan]Your thought")
            
            if not user_input.strip():
                continue
            
            # Handle commands
            if user_input.startswith('/'):
                result = await handle_command(engine, user_input)
                if result.get('exit'):
                    break
                continue
            
            # Process through CA/PD/MA pipeline
            console.print("[dim]Processing...[/dim]")
            
            with Live(console=console, refresh_per_second=2):
                response = await engine.process_user_input(user_input)
            
            # Display CA response
            console.print(Panel(
                Markdown(response['message']),
                title="Exploration Guide",
                border_style="blue"
            ))
            
            # Show insights if any
            if response.get('ma_insights'):
                console.print("[dim]💭 Meta-insights:[/dim]")
                for insight in response['ma_insights']:
                    console.print(f"  • {insight.get('observation', insight)}")
            
            # Show suggested commands
            if response.get('suggested_commands'):
                console.print(f"[dim]💡 Suggested: {', '.join(response['suggested_commands'])}[/dim]")
                
        except KeyboardInterrupt:
            console.print("\n[yellow]Session interrupted. Use /end to save properly.[/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


async def handle_command(engine: ExplorationEngine, command_input: str) -> dict:
    """Handle session commands."""
    parts = command_input[1:].split()  # Remove '/' and split
    command = parts[0] if parts else ""
    args = parts[1:] if len(parts) > 1 else []
    
    if command == "help":
        console.print(Panel(
            Markdown("""
# Session Commands

- `/map` - Show current session's concept map
- `/gapcheck` - Analyze gaps between your understanding and canonical knowledge  
- `/seed` - Get a new exploration seed from the Metacognitive Agent
- `/end` - End current session and get summary
- `/help` - Show this help message

Just type naturally to continue exploring!
            """),
            title="Help",
            border_style="cyan"
        ))
        return {}
    
    elif command in ["map", "gapcheck", "seed", "end"]:
        try:
            result = await engine.execute_command(command, args)
            
            if command == "end":
                display_session_summary(result)
                return {"exit": True}
            elif command == "map":
                display_session_map(result)
            elif command == "gapcheck":
                display_gap_check(result)
            elif command == "seed":
                display_seed(result)
                
        except Exception as e:
            console.print(f"[red]Command failed: {e}[/red]")
    
    else:
        console.print(f"[red]Unknown command: /{command}[/red]")
        console.print("Use /help to see available commands.")
    
    return {}


def display_session_map(map_data: dict):
    """Display session concept map."""
    if 'error' in map_data:
        console.print(f"[red]{map_data['error']}[/red]")
        return
    
    table = Table(title=f"Session Map: {map_data.get('topic', 'Unknown')}")
    table.add_column("Concept", style="cyan")
    table.add_column("Type", style="green") 
    table.add_column("Domain", style="yellow")
    
    for node in map_data.get('nodes', []):
        concept = node.get('label', 'Unknown')
        node_type = node.get('node_type', 'unknown')
        domain = node.get('properties', {}).get('domain', 'general')
        table.add_row(concept, node_type, domain)
    
    console.print(table)
    console.print(f"[dim]Connections made: {map_data.get('connections', 0)}[/dim]")


def display_gap_check(gap_data: dict):
    """Display understanding gap analysis."""
    console.print(Panel(
        Markdown(gap_data.get('message', 'Gap check analysis not available')),
        title="Understanding Gap Check",
        border_style="yellow"
    ))


def display_seed(seed_data: dict):
    """Display exploration seed."""
    if 'error' in seed_data:
        console.print(f"[red]{seed_data['error']}[/red]")
        return
        
    console.print(Panel(
        Markdown(f"""
# Exploration Seed: {seed_data.get('seed_concept', 'New Exploration')}

**Why this matters:** {seed_data.get('rationale', 'Continue exploring')}

**Questions to explore:**
{chr(10).join(f'• {q}' for q in seed_data.get('suggested_questions', []))}
        """),
        title="🌱 New Exploration Seed",
        border_style="green"
    ))


def display_session_summary(summary: dict):
    """Display session end summary."""
    duration = summary.get('duration_minutes', 0)
    
    console.print(Panel(
        Markdown(f"""
# Session Complete!

**Duration:** {duration:.1f} minutes  
**Exchanges:** {summary.get('total_exchanges', 0)}  
**Concepts Created:** {summary.get('concepts_created', 0)}  
**Connections Made:** {summary.get('connections_made', 0)}

**Session ID:** `{summary.get('session_id', 'unknown')}`

Use `peengine review <session_id>` to revisit this exploration later.
        """),
        title="✨ Session Summary",
        border_style="green"
    ))


@app.command()
def review(
    session_date: str = typer.Argument(..., help="Session date (YYYY-MM-DD) or session ID")
):
    """Review a past exploration session."""
    async def _review_session():
        engine = await initialize_engine()
        
        try:
            result = await engine.review_session(session_date)
            if result:
                console.print(f"[green]✓[/green] Session review for {session_date}")
                # Display review results
            else:
                console.print(f"[yellow]No session found for {session_date}[/yellow]")
                
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to review session: {e}")
    
    asyncio.run(_review_session())


@app.command()
def status(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging")
):
    """Show engine status and configuration."""
    async def _show_status():
        try:
            engine = await initialize_engine(verbose)
            
            table = Table(title="🧠 Personal Exploration Engine Status")
            table.add_column("Component", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Details")
            
            table.add_row("Database", "✓ Connected", "Neo4j ready")
            table.add_row("Conversational Agent", "✓ Ready", "Persona loaded")
            table.add_row("Pattern Detector", "✓ Ready", "Extraction templates loaded")
            table.add_row("Metacognitive Agent", "✓ Ready", "Analysis ready")
            
            console.print(table)
            
        except Exception as e:
            console.print(f"[red]Engine not ready: {e}[/red]")
    
    asyncio.run(_show_status())


@app.command()
def init(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging")
):
    """Initialize configuration and check dependencies."""
    console.print("🔧 Initializing Personal Exploration Engine...")
    
    # Check for .env file
    if not Path(".env").exists():
        console.print("[yellow]⚠[/yellow] .env file not found")
        console.print("Copy .env.example to .env and configure your settings")
        return
    
    # Try to load settings
    try:
        settings = Settings()
        console.print(f"[green]✓[/green] Configuration loaded")
        console.print(f"  • Persona path: {settings.persona_path}")
        console.print(f"  • Knowledge graphs: {settings.knowledge_graphs_path}")
        console.print(f"  • Neo4j URI: {settings.neo4j_uri}")
    except Exception as e:
        console.print(f"[red]✗[/red] Configuration error: {e}")
        return
    
    console.print("\n[green]✓[/green] Ready to explore! Use 'peengine start <topic>' to begin.")


@app.command(name="import-graphs")
def import_knowledge_graphs(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging")
):
    """Import existing knowledge graphs from JSON files into Neo4j."""
    async def _import_graphs():
        console.print("🧠 Importing existing knowledge graphs...")
        
        try:
            settings = Settings()
            # Only setup logging if verbose mode is enabled
            if verbose:
                setup_logging(settings.log_level)
            else:
                setup_logging("ERROR")
            
            importer = KnowledgeGraphImporter(settings)
            results = await importer.import_all_graphs()
            
            # Display results
            console.print(f"[green]✅ Import complete![/green]")
            
            # Create results table
            table = Table(title="Import Results")
            table.add_column("Metric", style="cyan")
            table.add_column("Count", style="green")
            
            table.add_row("Domains imported", str(results['domains_imported']))
            table.add_row("Connections imported", str(results['connections_imported']))
            table.add_row("Total nodes created", str(results['total_nodes']))
            table.add_row("Total edges created", str(results['total_edges']))
            
            console.print(table)
            
            # Show errors if any
            if results['errors']:
                console.print(f"\n[yellow]⚠️  Errors encountered:[/yellow]")
                for error in results['errors']:
                    console.print(f"  • [red]{error}[/red]")
            
            if results['total_nodes'] > 0:
                console.print(f"\n[green]✓[/green] Successfully imported {results['total_nodes']} concepts into your knowledge graph!")
                console.print("Use 'peengine start <topic>' to explore with your imported knowledge.")
            else:
                console.print(f"\n[yellow]⚠[/yellow] No knowledge graphs found to import.")
                console.print(f"Make sure you have JSON files in: {settings.knowledge_graphs_path}")
                
        except Exception as e:
            console.print(f"[red]✗[/red] Import failed: {e}")
            raise typer.Exit(1)
    
    asyncio.run(_import_graphs())


@app.command(name="start-fresh")  
def start_fresh(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging")
):
    """Clear all databases and start completely fresh. DESTRUCTIVE OPERATION!"""
    async def _clear_databases():
        try:
            settings = Settings()
            # Only setup logging if verbose mode is enabled
            if verbose:
                setup_logging(settings.log_level)
            else:
                setup_logging("ERROR")
            
            from .tools.clear_databases import DatabaseCleaner
            cleaner = DatabaseCleaner(settings)
            
            result = await cleaner.clear_all_databases()
            
            if result.get("cancelled"):
                console.print("[yellow]Operation cancelled by user.[/yellow]")
                return  # Exit cleanly without error
            elif not result.get("success"):
                console.print("[red]Database clearing completed with errors.[/red]")
                raise typer.Exit(1)
            else:
                console.print("[green]Database clearing completed successfully.[/green]")
                return  # Exit cleanly without error
                
        except Exception as e:
            console.print(f"[red]✗[/red] Database clearing failed: {e}")
            raise typer.Exit(1)
    
    asyncio.run(_clear_databases())


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()