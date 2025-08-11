"""CLI interface for Zerdisha."""

from .tools.import_knowledge_graphs import KnowledgeGraphImporter
from .core.orchestrator import ExplorationEngine
from .models.config import Settings
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt
from rich.live import Live
from rich.markdown import Markdown

# Setup logger
logger = logging.getLogger(__name__)


# Initialize Typer app and Rich console
app = typer.Typer(
    name="zerdisha",
    help="Zerdisha - A terminal-based interactive learning tool"
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
    title: Optional[str] = typer.Option(
        None, "--title", "-t", help="Session title"),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging")
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
- `/analyze` - Run Metacognitive analysis (no changes applied)
- `/apply_ma` - Apply last Metacognitive persona adjustments
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
                    insight_text = insight.get('observation', insight) if isinstance(
                        insight, dict) else str(insight)
                    console.print(f"  • {insight_text}")

            # Show suggested commands
            if response.get('suggested_commands'):
                console.print(
                    f"[dim]💡 Suggested: {', '.join(response['suggested_commands'])}[/dim]")

        except KeyboardInterrupt:
            console.print(
                "\n[yellow]Session interrupted. Use /end to save properly.[/yellow]")
            break
        except Exception as e:
            # Use adaptive error recovery for conversation processing
            error_context = {
                "error_type": "conversation_processing",
                "error": str(e),
                "user_input": user_input,
                "session_active": True,
                "session_topic": engine.current_session.topic if engine.current_session else None
            }

            recovery_message = await generate_adaptive_conversation_error(engine, error_context)
            console.print(Panel(
                Markdown(recovery_message),
                title="⚠️ Processing Issue",
                border_style="yellow"
            ))


async def handle_command(engine: ExplorationEngine, command_input: str) -> dict:
    """Handle session commands with adaptive error recovery."""
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
- `/analyze` - Run Metacognitive analysis (command-only)
- `/apply_ma` - Apply the last MA persona adjustments (if any)
- `/end` - End current session and get summary
- `/help` - Show this help message

Just type naturally to continue exploring!
            """),
            title="Help",
            border_style="cyan"
        ))
        return {}

    elif command in ["map", "gapcheck", "seed", "end", "analyze", "apply_ma"]:
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
            elif command == "analyze":
                display_analysis(result)
            elif command == "apply_ma":
                display_apply_ma(result)

        except Exception as e:
            # Use adaptive error recovery
            error_context = {
                "command": command,
                "args": args,
                "error": str(e),
                "session_active": engine.current_session is not None,
                "session_topic": engine.current_session.topic if engine.current_session else None
            }

            recovery_message = await generate_adaptive_error_message(engine, error_context)
            console.print(Panel(
                Markdown(recovery_message),
                title="⚠️ Command Issue",
                border_style="yellow"
            ))

    else:
        console.print(f"[red]Unknown command: /{command}[/red]")
        console.print("Use /help to see available commands.")

    return {}


async def generate_adaptive_conversation_error(engine: ExplorationEngine, error_context: Dict[str, Any]) -> str:
    """Generate context-aware error messages for conversation processing failures."""

    error = error_context.get('error', 'Unknown error')
    user_input = error_context.get('user_input', 'your input')
    session_topic = error_context.get('session_topic', 'Unknown')

    prompt = f"""
A learner encountered an issue while exploring "{session_topic}". They said: "{user_input[:100]}..."

The system error was: {error}

Create a helpful, encouraging response that:
1. Acknowledges their input and shows you understood their intent
2. Explains the issue in simple, non-technical terms
3. Suggests how to continue the exploration
4. Maintains the Socratic learning atmosphere
5. Offers alternative ways to explore their question

Keep it warm, supportive, and focused on continuing their learning journey.
"""

    try:
        response = await engine.ca.client.chat.completions.create(
            model=engine.ca._get_deployment_name(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=200
        )

        return response.choices[0].message.content.strip()

    except Exception as llm_error:
        logger.error(
            f"Failed to generate adaptive conversation error: {llm_error}")
        return f"""
I can see you're exploring interesting ideas about {session_topic}, but I encountered a technical hiccup processing your thought.

**Let's keep the exploration going:**
• Try rephrasing your question or observation
• We can approach this topic from a different angle
• Your curiosity is valuable - don't let this pause stop you!

What aspect of {session_topic} would you like to explore next?
"""


async def generate_adaptive_error_message(engine: ExplorationEngine, error_context: Dict[str, Any]) -> str:
    """Generate context-aware error messages and recovery suggestions using LLM."""

    command = error_context.get('command', 'unknown')
    error = error_context.get('error', 'Unknown error')
    session_active = error_context.get('session_active', False)
    session_topic = error_context.get('session_topic', 'Unknown')

    # Get recent conversation context if available
    recent_context = ""
    if engine.current_session and engine.current_session.messages:
        # Last 2 exchanges
        recent_messages = engine.current_session.messages[-2:]
        recent_context = "\n".join([
            f"User: {msg.get('user', '')}\nAssistant: {msg.get('assistant', '')}"
            for msg in recent_messages
        ])

    prompt = f"""
You are helping a learner who encountered an issue while using a learning exploration tool. Provide a helpful, encouraging error message with specific recovery suggestions.

COMMAND THAT FAILED: /{command}
ERROR DETAILS: {error}
SESSION ACTIVE: {session_active}
SESSION TOPIC: {session_topic}

RECENT CONVERSATION:
{recent_context}

Create a helpful error message that:
1. Acknowledges what they were trying to do
2. Explains what went wrong in simple terms
3. Provides specific steps to recover or work around the issue
4. Maintains an encouraging, supportive tone
5. Suggests alternative approaches if the main command isn't working
6. Relates to their current exploration context

Keep it concise but helpful. Use a warm, understanding tone that fits the Socratic learning environment.

Example format:
I understand you were trying to [what they wanted to do]. It looks like [simple explanation of issue]. 

Here's how we can get back on track:
• [specific recovery step 1]
• [specific recovery step 2]
• [alternative approach if needed]

Your exploration of {session_topic} is going well - let's keep the momentum going!
"""

    try:
        response = await engine.ca.client.chat.completions.create(
            model=engine.ca._get_deployment_name(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,  # Balanced helpfulness and consistency
            max_tokens=250
        )

        adaptive_message = response.choices[0].message.content.strip()
        logger.info(f"Generated adaptive error message for /{command} failure")
        return adaptive_message

    except Exception as llm_error:
        logger.error(f"Failed to generate adaptive error message: {llm_error}")
        # Fallback to simple error message
        return f"""
I encountered an issue with the `/{command}` command: {error}

**Quick fixes to try:**
• Make sure you have an active session running
• Check your internet connection for AI services
• Try the command again in a moment
• Use `/help` to see available commands

Don't worry - your exploration progress is saved! Let's keep going.
"""


def display_session_map(map_data: dict):
    """Display session concept map with comprehensive error handling and fallback mechanisms."""

    # Handle error cases with helpful messages
    if 'error' in map_data:
        # Tests expect a single red-line error output for session map errors
        error_message = map_data.get('error')
        console.print(f"[red]{error_message}[/red]")
        return

    # Handle empty session case
    if map_data.get('status') == 'empty_session':
        console.print(Panel(
            Markdown(map_data.get('message', 'No concepts mapped yet')),
            title=f"🗺️ Session Map: {map_data.get('topic', 'Unknown')}",
            border_style="blue"
        ))
        return

    # Display nodes table with error handling
    try:
        nodes = map_data.get('nodes', [])
        topic = map_data.get('topic', 'Unknown')

        if nodes:
            nodes_table = Table(title=f"Session Map: {topic}")
            nodes_table.add_column("Concept", style="cyan")
            nodes_table.add_column("Type", style="green")
            nodes_table.add_column("Domain", style="yellow")

            for node in nodes:
                try:
                    concept = node.get('label', 'Unknown')
                    node_type = node.get('node_type', 'unknown')
                    properties = node.get('properties') or {}
                    domain = properties.get('domain', 'general')
                    nodes_table.add_row(concept, node_type, domain)
                except Exception as node_error:
                    logger.warning(f"Error displaying node: {node_error}")
                    nodes_table.add_row(
                        "Error loading concept", "unknown", "unknown")

            console.print(nodes_table)
        else:
            console.print(f"[dim]📍 Session: {topic}[/dim]")
            console.print("[yellow]No concepts to display[/yellow]")

    except Exception as table_error:
        logger.error(f"Error creating nodes table: {table_error}")
        console.print("[red]Error displaying concept table[/red]")

    # Display relationships with comprehensive error handling
    try:
        edges = map_data.get('edges', [])
        relationship_descriptions = map_data.get(
            'relationship_descriptions', [])

        if edges:
            console.print("\n[bold]Conceptual Connections:[/bold]")

            # Try to use LLM-generated descriptions first
            if relationship_descriptions:
                try:
                    for desc in relationship_descriptions:
                        if desc and isinstance(desc, str):
                            console.print(desc)
                        else:
                            console.print(
                                "• Connection description unavailable")
                except Exception as desc_error:
                    logger.warning(
                        f"Error displaying relationship descriptions: {desc_error}")
                    # Fall back to simple format
                    display_simple_relationships(edges)
            else:
                # Fallback to simple table format
                display_simple_relationships(edges)
        else:
            if nodes:  # Only show this if we have nodes but no edges
                console.print("\n[dim]🔗 No connections mapped yet[/dim]")

    except Exception as relationships_error:
        logger.error(f"Error displaying relationships: {relationships_error}")
        console.print("[red]Error displaying connections[/red]")

    # Display summary statistics with error handling
    try:
        node_count = map_data.get('node_count', len(map_data.get('nodes', [])))
        connection_count = map_data.get(
            'connection_count', len(map_data.get('edges', [])))

        console.print(
            f"\n[dim]📊 Summary: {node_count} concepts, {connection_count} connections[/dim]")

        # Display warnings if there were data issues
        if 'warnings' in map_data:
            console.print(f"\n[yellow]⚠️  Data issues:[/yellow]")
            for warning in map_data['warnings']:
                console.print(f"  • {warning}")

    except Exception as summary_error:
        logger.error(f"Error displaying summary: {summary_error}")
        console.print("[dim]Summary unavailable[/dim]")


def display_simple_relationships(edges: list):
    """Display relationships in simple table format as fallback."""
    try:
        relationships_table = Table()
        relationships_table.add_column("Connection", style="magenta", width=60)
        relationships_table.add_column("Type", style="blue")

        for edge in edges:
            try:
                source_label = edge.get('source_label', 'Unknown')
                target_label = edge.get('target_label', 'Unknown')
                edge_type = edge.get('edge_type', 'relates')

                # Format relationship in human-readable form (escape brackets for Rich)
                connection_text = f"\\[{source_label}] --({edge_type})--> \\[{target_label}]"
                relationships_table.add_row(connection_text, edge_type)

            except Exception as edge_error:
                logger.warning(f"Error displaying edge: {edge_error}")
                relationships_table.add_row(
                    "Connection unavailable", "unknown")

        console.print(relationships_table)

    except Exception as table_error:
        logger.error(f"Error creating relationships table: {table_error}")
        console.print("[red]Error displaying connection table[/red]")
    # Summary is printed by display_session_map; no summary here


def display_gap_check(gap_data: dict):
    """Display understanding gap analysis with comprehensive error handling."""

    # Handle error cases
    if 'error' in gap_data:
        # Tests expect a standardized fallback message in a Panel for gap check errors
        error_message = 'Gap check analysis not available'
        console.print(Panel(
            Markdown(error_message),
            title="🔍 Understanding Gap Check",
            border_style="yellow"
        ))

        # Show recovery suggestions if available
        if 'recovery_suggestions' in gap_data:
            console.print("\n[dim]💡 Try these steps:[/dim]")
            for suggestion in gap_data['recovery_suggestions']:
                console.print(f"  • {suggestion}")
        return

    # Handle successful gap analysis
    try:
        message = gap_data.get('message', 'Gap check analysis not available')

        # Determine border color based on gap severity
        severity = gap_data.get('severity', 'unknown')
        if severity in ['minimal', 'low']:
            border_style = "green"
        elif severity == 'moderate':
            border_style = "yellow"
        else:
            border_style = "red"

        console.print(Panel(
            Markdown(message),
            title="🔍 Understanding Gap Check",
            border_style=border_style
        ))

        # Display technical details if available (for debugging/advanced users)
        if gap_data.get('success') and any(key in gap_data for key in ['similarity', 'gap_score', 'concept']):
            concept = gap_data.get('concept', 'Unknown')
            similarity = gap_data.get('similarity', 0.0)

            console.print(
                f"\n[dim]📊 Analysis: {concept} | Similarity: {similarity:.2f} | Severity: {severity}[/dim]")

    except Exception as display_error:
        logger.error(f"Error displaying gap check: {display_error}")
        console.print(Panel(
            Markdown(
                "⚠️ **Gap analysis display error**\n\nThere was an issue displaying your gap analysis results."),
            title="🔍 Understanding Gap Check",
            border_style="red"
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


def display_analysis(result: dict):
    """Display MA analysis results (command-only)."""
    if 'error' in result:
        console.print(f"[red]{result['error']}[/red]")
        return

    analysis = result.get('analysis', {}) or {}
    applied = result.get('applied', False)
    insights = analysis.get('insights', [])
    flags = analysis.get('flags', [])
    has_adjustments = bool(analysis.get('persona_adjustments'))

    md = ["# Meta Analysis Results\n"]
    if insights:
        md.append("**Insights:**")
        md.extend([f"• {i}" for i in insights])
    if flags:
        md.append("\n**Flags:**")
        md.extend([f"• {f}" for f in flags])
    md.append(
        f"\n**Persona adjustments available:** {'Yes' if has_adjustments else 'No'}")
    md.append("\nRun `/apply_ma` to apply if available.")

    console.print(Panel(Markdown("\n".join(md)),
                  title="🧠 Metacognitive Analysis", border_style="cyan"))


def display_apply_ma(result: dict):
    """Display the result of applying MA persona adjustments."""
    if 'error' in result:
        console.print(f"[red]{result['error']}[/red]")
        return

    if result.get('applied'):
        adjustments = result.get('adjustments', {})
        lines = ["# Persona Adjustments Applied\n"]
        for k, v in (adjustments or {}).items():
            lines.append(f"• {k}: {v}")
        console.print(Panel(Markdown("\n".join(lines)),
                      title="✅ Applied", border_style="green"))
    else:
        console.print(Panel(Markdown("No persona adjustments to apply."),
                      title="ℹ️ Nothing to Apply", border_style="yellow"))


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
    session_date: str = typer.Argument(...,
                                       help="Session date (YYYY-MM-DD) or session ID")
):
    """Review a past exploration session."""
    async def _review_session():
        engine = await initialize_engine()

        try:
            result = await engine.review_session(session_date)
            if result:
                console.print(
                    f"[green]✓[/green] Session review for {session_date}")
                # Display review results
            else:
                console.print(
                    f"[yellow]No session found for {session_date}[/yellow]")

        except Exception as e:
            console.print(f"[red]✗[/red] Failed to review session: {e}")

    asyncio.run(_review_session())


@app.command()
def status(
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging")
):
    """Show engine status and configuration."""
    async def _show_status():
        try:
            engine = await initialize_engine(verbose)

            table = Table(title="🧠 Zerdisha Status")
            table.add_column("Component", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Details")

            table.add_row("Database", "✓ Connected", "Neo4j ready")
            table.add_row("Conversational Agent", "✓ Ready", "Persona loaded")
            table.add_row("Pattern Detector", "✓ Ready",
                          "Extraction templates loaded")
            table.add_row("Metacognitive Agent", "✓ Ready", "Analysis ready")

            console.print(table)

        except Exception as e:
            console.print(f"[red]Engine not ready: {e}[/red]")

    asyncio.run(_show_status())


@app.command()
def init(
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging")
):
    """Initialize configuration and check dependencies."""
    console.print("🔧 Initializing Zerdisha...")

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
        console.print(
            f"  • Knowledge graphs: {settings.knowledge_graphs_path}")
        console.print(f"  • Neo4j URI: {settings.neo4j_uri}")
    except Exception as e:
        console.print(f"[red]✗[/red] Configuration error: {e}")
        return

    console.print(
        "\n[green]✓[/green] Ready to explore! Use 'zerdisha start <topic>' to begin.")


@app.command(name="import-graphs")
def import_knowledge_graphs(
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging")
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
            table.add_row("Connections imported", str(
                results['connections_imported']))
            table.add_row("Total nodes created", str(results['total_nodes']))
            table.add_row("Total edges created", str(results['total_edges']))

            console.print(table)

            # Show errors if any
            if results['errors']:
                console.print(f"\n[yellow]⚠️  Errors encountered:[/yellow]")
                for error in results['errors']:
                    console.print(f"  • [red]{error}[/red]")

            if results['total_nodes'] > 0:
                console.print(
                    f"\n[green]✓[/green] Successfully imported {results['total_nodes']} concepts into your knowledge graph!")
                console.print(
                    "Use 'zerdisha start <topic>' to explore with your imported knowledge.")
            else:
                console.print(
                    f"\n[yellow]⚠[/yellow] No knowledge graphs found to import.")
                console.print(
                    f"Make sure you have JSON files in: {settings.knowledge_graphs_path}")

        except Exception as e:
            console.print(f"[red]✗[/red] Import failed: {e}")
            raise typer.Exit(1)

    asyncio.run(_import_graphs())


@app.command(name="start-fresh")
def start_fresh(
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging")
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
                console.print(
                    "[red]Database clearing completed with errors.[/red]")
                raise typer.Exit(1)
            else:
                console.print(
                    "[green]Database clearing completed successfully.[/green]")
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
